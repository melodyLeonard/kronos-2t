//! Memory-mapped device context index, learned device profile artifact, and macOS computer control.
//!
//! Stores a compact binary index (`device_index.k2di`), learned facts (`learned_facts.json`),
//! and a human-readable summary (`device_profile.md`) under `~/.kronos/device_context/`.
//! Retrieval runs against in-RAM FNV-1a word/acronym and ASCII 3-gram posting lists in sub-millisecond time.

use crate::json::{quote, Json};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const K2DI_MAGIC: [u8; 8] = *b"K2DI\x01\x00\x00\x00";
const HEADER_BYTES: usize = 32;
const PACKED_ENTRY_BYTES: usize = 44;
const MAX_LEARNED_FACTS: usize = 256;

const FLAG_EXISTS_ON_DISK: u8 = 1 << 0;
const FLAG_WATCHED_HISTORY: u8 = 1 << 1;
const FLAG_SERIES_EPISODE: u8 = 1 << 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EntryKind {
    App = 1,
    Document = 2,
    Media = 3,
    Project = 4,
    Directory = 5,
}

impl EntryKind {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => EntryKind::App,
            2 => EntryKind::Document,
            3 => EntryKind::Media,
            4 => EntryKind::Project,
            _ => EntryKind::Directory,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            EntryKind::App => "app",
            EntryKind::Document => "document",
            EntryKind::Media => "media",
            EntryKind::Project => "project",
            EntryKind::Directory => "folder",
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviceConfig {
    pub enabled: bool,
    pub allow_computer_control: bool,
    pub updated_at: u64,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allow_computer_control: true,
            updated_at: now_epoch_secs(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LearnedFact {
    pub key: String,
    pub value: String,
    pub path: String,
    pub updated_at: u64,
    pub hits: u32,
}

#[derive(Clone, Debug)]
struct PackedMeta {
    kind: EntryKind,
    flags: u8,
    mru_rank: u16,
    name_off: u32,
    name_len: u32,
    path_off: u32,
    path_len: u32,
    detail_off: u32,
    detail_len: u32,
    modified_epoch: u64,
    size_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct DeviceHit {
    pub kind: EntryKind,
    pub name: String,
    pub path: String,
    pub detail: String,
    pub exists_on_disk: bool,
    pub watched_history: bool,
    pub is_series_episode: bool,
    pub mru_rank: u16,
    pub modified_epoch: u64,
    pub size_bytes: u64,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct DeviceTurnContext {
    pub prompt_block: String,
    pub hits: Vec<DeviceHit>,
    pub elapsed_ms: f64,
    pub wants_computer_action: bool,
}

#[derive(Clone, Debug)]
pub struct DeviceStatus {
    pub enabled: bool,
    pub allow_computer_control: bool,
    pub indexing: bool,
    pub indexed_entries: usize,
    pub apps_count: usize,
    pub docs_count: usize,
    pub media_count: usize,
    pub projects_count: usize,
    pub learned_facts_count: usize,
    pub build_ms: f64,
    pub artifact_bytes: usize,
    pub artifact_path: String,
    pub profile_path: String,
    pub built_at: u64,
}

pub struct DeviceIndex {
    built_at: u64,
    build_micros: u64,
    artifact_bytes: usize,
    entries: Vec<PackedMeta>,
    arena: Vec<u8>,
    word_postings: HashMap<u32, Vec<u32>>,
    trigram_postings: HashMap<u32, Vec<u32>>,
}

impl Default for DeviceIndex {
    fn default() -> Self {
        Self::empty()
    }
}

impl DeviceIndex {
    pub fn empty() -> Self {
        Self {
            built_at: 0,
            build_micros: 0,
            artifact_bytes: 0,
            entries: Vec::new(),
            arena: Vec::new(),
            word_postings: HashMap::new(),
            trigram_postings: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn str_slice(&self, off: u32, len: u32) -> &str {
        let s = off as usize;
        let e = s.saturating_add(len as usize);
        if e <= self.arena.len() {
            std::str::from_utf8(&self.arena[s..e]).unwrap_or("")
        } else {
            ""
        }
    }

    fn entry_to_hit(&self, idx: usize, score: f32) -> DeviceHit {
        let m = &self.entries[idx];
        DeviceHit {
            kind: m.kind,
            name: self.str_slice(m.name_off, m.name_len).to_string(),
            path: self.str_slice(m.path_off, m.path_len).to_string(),
            detail: self.str_slice(m.detail_off, m.detail_len).to_string(),
            exists_on_disk: (m.flags & FLAG_EXISTS_ON_DISK) != 0,
            watched_history: (m.flags & FLAG_WATCHED_HISTORY) != 0,
            is_series_episode: (m.flags & FLAG_SERIES_EPISODE) != 0,
            mru_rank: m.mru_rank,
            modified_epoch: m.modified_epoch,
            size_bytes: m.size_bytes,
            score,
        }
    }

    fn rebuild_postings(&mut self) {
        let mut words: HashMap<u32, Vec<u32>> = HashMap::with_capacity(self.entries.len() * 3);
        let mut trigrams: HashMap<u32, Vec<u32>> = HashMap::with_capacity(self.entries.len() * 6);

        for idx in 0..self.entries.len() {
            let id = idx as u32;
            let m = &self.entries[idx];
            let name = self.str_slice(m.name_off, m.name_len);
            let detail = self.str_slice(m.detail_off, m.detail_len);
            let path = self.str_slice(m.path_off, m.path_len);
            let parent_name = Path::new(path)
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("");

            let combined = format!("{name} {detail} {parent_name}");
            let toks = tokenize_words_with_acronyms(&combined);
            for tok in &toks {
                let h = fnv1a_u32(tok.as_bytes());
                let list = words.entry(h).or_default();
                if list.last().copied() != Some(id) {
                    list.push(id);
                }
                if tok.len() >= 3 {
                    for tg in extract_trigrams(tok) {
                        let tlist = trigrams.entry(tg).or_default();
                        if tlist.last().copied() != Some(id) {
                            tlist.push(id);
                        }
                    }
                }
            }
        }
        self.word_postings = words;
        self.trigram_postings = trigrams;
    }

    pub fn to_k2di_bytes(&self) -> Vec<u8> {
        let total = HEADER_BYTES + self.entries.len() * PACKED_ENTRY_BYTES + self.arena.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&K2DI_MAGIC);
        out.extend_from_slice(&self.built_at.to_le_bytes());
        out.extend_from_slice(&self.build_micros.to_le_bytes());
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.arena.len() as u32).to_le_bytes());

        for m in &self.entries {
            out.push(m.kind as u8);
            out.push(m.flags);
            out.extend_from_slice(&m.mru_rank.to_le_bytes());
            out.extend_from_slice(&m.name_off.to_le_bytes());
            out.extend_from_slice(&m.name_len.to_le_bytes());
            out.extend_from_slice(&m.path_off.to_le_bytes());
            out.extend_from_slice(&m.path_len.to_le_bytes());
            out.extend_from_slice(&m.detail_off.to_le_bytes());
            out.extend_from_slice(&m.detail_len.to_le_bytes());
            out.extend_from_slice(&m.modified_epoch.to_le_bytes());
            out.extend_from_slice(&m.size_bytes.to_le_bytes());
        }
        out.extend_from_slice(&self.arena);
        out
    }

    pub fn from_k2di_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_BYTES || bytes[..8] != K2DI_MAGIC {
            return Err("invalid k2di magic header".to_string());
        }
        let built_at = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let build_micros = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let entry_count = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
        let arena_len = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;
        let entries_end = HEADER_BYTES
            .checked_add(entry_count.saturating_mul(PACKED_ENTRY_BYTES))
            .ok_or("k2di entry table overflow")?;
        let total_end = entries_end.checked_add(arena_len).ok_or("k2di arena overflow")?;
        if bytes.len() < total_end {
            return Err("truncated k2di file".to_string());
        }

        let mut entries = Vec::with_capacity(entry_count);
        for i in 0..entry_count {
            let base = HEADER_BYTES + i * PACKED_ENTRY_BYTES;
            let rec = &bytes[base..base + PACKED_ENTRY_BYTES];
            entries.push(PackedMeta {
                kind: EntryKind::from_u8(rec[0]),
                flags: rec[1],
                mru_rank: u16::from_le_bytes(rec[2..4].try_into().unwrap()),
                name_off: u32::from_le_bytes(rec[4..8].try_into().unwrap()),
                name_len: u32::from_le_bytes(rec[8..12].try_into().unwrap()),
                path_off: u32::from_le_bytes(rec[12..16].try_into().unwrap()),
                path_len: u32::from_le_bytes(rec[16..20].try_into().unwrap()),
                detail_off: u32::from_le_bytes(rec[20..24].try_into().unwrap()),
                detail_len: u32::from_le_bytes(rec[24..28].try_into().unwrap()),
                modified_epoch: u64::from_le_bytes(rec[28..36].try_into().unwrap()),
                size_bytes: u64::from_le_bytes(rec[36..44].try_into().unwrap()),
            });
        }
        let arena = bytes[entries_end..total_end].to_vec();
        let mut idx = Self {
            built_at,
            build_micros,
            artifact_bytes: bytes.len(),
            entries,
            arena,
            word_postings: HashMap::new(),
            trigram_postings: HashMap::new(),
        };
        idx.rebuild_postings();
        Ok(idx)
    }

    fn counts_by_kind(&self) -> (usize, usize, usize, usize) {
        let (mut apps, mut docs, mut media, mut projs) = (0, 0, 0, 0);
        for e in &self.entries {
            match e.kind {
                EntryKind::App => apps += 1,
                EntryKind::Document => docs += 1,
                EntryKind::Media => media += 1,
                EntryKind::Project => projs += 1,
                EntryKind::Directory => {}
            }
        }
        (apps, docs, media, projs)
    }

    pub fn search(
        &self,
        raw_query: &str,
        facts: &[LearnedFact],
        kind_filter: Option<EntryKind>,
        limit: usize,
    ) -> Vec<DeviceHit> {
        let lower = raw_query.to_ascii_lowercase();
        let query_tokens = extract_meaningful_query_tokens(&lower);
        let synonyms = expand_query_synonyms(&query_tokens);
        let wants_media = is_media_history_query(&lower);
        let wants_app = is_app_action_query(&lower);

        let mut scores: HashMap<u32, f32> = HashMap::new();

        for tok in &query_tokens {
            let h = fnv1a_u32(tok.as_bytes());
            let exact_weight = if tok.len() <= 3 { 78.0 } else { 58.0 };
            if let Some(ids) = self.word_postings.get(&h) {
                let idf_scale = (1.0 + (40.0 / (ids.len() as f32 + 1.0)).ln()).clamp(1.0, 2.2);
                for &id in ids {
                    *scores.entry(id).or_insert(0.0) += exact_weight * idf_scale;
                }
            }

            if tok.len() >= 4 {
                let tgs = extract_trigrams(tok);
                if !tgs.is_empty() {
                    let mut tg_counts: HashMap<u32, u32> = HashMap::new();
                    for tg in &tgs {
                        if let Some(ids) = self.trigram_postings.get(tg) {
                            for &id in ids {
                                *tg_counts.entry(id).or_insert(0) += 1;
                            }
                        }
                    }
                    let denom = tgs.len() as f32;
                    for (id, matched) in tg_counts {
                        let ratio = matched as f32 / denom;
                        if ratio >= 0.42 {
                            *scores.entry(id).or_insert(0.0) += 42.0 * ratio;
                        }
                    }
                }
            }
        }

        for syn in &synonyms {
            let h = fnv1a_u32(syn.as_bytes());
            if let Some(ids) = self.word_postings.get(&h) {
                for &id in ids {
                    *scores.entry(id).or_insert(0.0) += 26.0;
                }
            }
        }

        // Boost entries matching learned facts.
        if !facts.is_empty() && !query_tokens.is_empty() {
            let mut path_boosts: HashMap<&str, f32> = HashMap::new();
            for fact in facts {
                let fk = fact.key.to_ascii_lowercase();
                let fv = fact.value.to_ascii_lowercase();
                let matched = query_tokens
                    .iter()
                    .filter(|t| fk.contains(t.as_str()) || fv.contains(t.as_str()))
                    .count();
                if matched > 0 && !fact.path.is_empty() {
                    let bonus = 90.0 + (matched as f32 * 25.0) + (fact.hits.min(10) as f32 * 3.0);
                    path_boosts.insert(fact.path.as_str(), bonus);
                }
            }
            if !path_boosts.is_empty() {
                for (idx, m) in self.entries.iter().enumerate() {
                    let p = self.str_slice(m.path_off, m.path_len);
                    if let Some(&b) = path_boosts.get(p) {
                        *scores.entry(idx as u32).or_insert(0.0) += b;
                    }
                }
            }
        }

        // Pure recency media query ("what movie did I watch last") when query tokens are generic.
        if wants_media {
            for (idx, m) in self.entries.iter().enumerate() {
                if m.kind == EntryKind::Media && (m.flags & FLAG_WATCHED_HISTORY) != 0 && m.mru_rank > 0 {
                    let recency = (120.0 - (m.mru_rank as f32 * 2.5)).max(15.0);
                    let movie_bonus = if (lower.contains("movie") || lower.contains("film"))
                        && (m.flags & FLAG_SERIES_EPISODE) == 0
                    {
                        28.0
                    } else {
                        0.0
                    };
                    *scores.entry(idx as u32).or_insert(0.0) += recency + movie_bonus;
                }
            }
        }

        let mut hits: Vec<DeviceHit> = Vec::with_capacity(scores.len().min(64));
        for (id, mut sc) in scores {
            let idx = id as usize;
            if idx >= self.entries.len() {
                continue;
            }
            let m = &self.entries[idx];
            if let Some(kf) = kind_filter {
                if m.kind != kf {
                    continue;
                }
            }
            if wants_app && m.kind == EntryKind::App {
                sc += 48.0;
                let app_name = self.str_slice(m.name_off, m.name_len).to_ascii_lowercase();
                if query_tokens.contains(&app_name) {
                    sc += 65.0;
                } else if query_tokens.iter().any(|t| app_name.starts_with(t.as_str())) {
                    sc += 32.0;
                }
            }
            if !wants_app && !wants_media && m.kind == EntryKind::Document {
                sc += 14.0;
            }
            if (m.flags & FLAG_EXISTS_ON_DISK) != 0 {
                sc += 6.0;
            }
            hits.push(self.entry_to_hit(idx, sc));
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.modified_epoch.cmp(&a.modified_epoch))
        });
        hits.truncate(limit);
        hits
    }

    pub fn recent_watched_media(&self, limit: usize) -> Vec<DeviceHit> {
        let mut items: Vec<DeviceHit> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, m)| m.kind == EntryKind::Media && (m.flags & FLAG_WATCHED_HISTORY) != 0)
            .map(|(i, m)| {
                let score = 1000.0 - m.mru_rank as f32;
                self.entry_to_hit(i, score)
            })
            .collect();
        items.sort_by(|a, b| {
            a.mru_rank
                .cmp(&b.mru_rank)
                .then_with(|| b.modified_epoch.cmp(&a.modified_epoch))
        });
        items.truncate(limit);
        items
    }
}

struct IndexBuilder {
    entries: Vec<PackedMeta>,
    arena: Vec<u8>,
    seen_paths: HashMap<String, usize>,
}

impl IndexBuilder {
    fn new() -> Self {
        Self {
            entries: Vec::with_capacity(1024),
            arena: Vec::with_capacity(96 * 1024),
            seen_paths: HashMap::with_capacity(1024),
        }
    }

    fn push_str(&mut self, s: &str) -> (u32, u32) {
        let off = self.arena.len() as u32;
        self.arena.extend_from_slice(s.as_bytes());
        (off, s.len() as u32)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_entry(
        &mut self,
        kind: EntryKind,
        name: &str,
        path: &str,
        detail: &str,
        flags: u8,
        mru_rank: u16,
        modified_epoch: u64,
        size_bytes: u64,
    ) {
        if name.is_empty() || path.is_empty() {
            return;
        }
        if let Some(&existing_idx) = self.seen_paths.get(path) {
            if (flags & FLAG_WATCHED_HISTORY) != 0 {
                let (d_off, d_len) = self.push_str(detail);
                let e = &mut self.entries[existing_idx];
                e.flags |= flags;
                if mru_rank > 0 && (e.mru_rank == 0 || mru_rank < e.mru_rank) {
                    e.mru_rank = mru_rank;
                }
                e.detail_off = d_off;
                e.detail_len = d_len;
            }
            return;
        }
        let (name_off, name_len) = self.push_str(name);
        let (path_off, path_len) = self.push_str(path);
        let (detail_off, detail_len) = self.push_str(detail);
        let idx = self.entries.len();
        self.seen_paths.insert(path.to_string(), idx);
        self.entries.push(PackedMeta {
            kind,
            flags,
            mru_rank,
            name_off,
            name_len,
            path_off,
            path_len,
            detail_off,
            detail_len,
            modified_epoch,
            size_bytes,
        });
    }

    fn finish(self, build_micros: u64) -> DeviceIndex {
        let mut idx = DeviceIndex {
            built_at: now_epoch_secs(),
            build_micros,
            artifact_bytes: 0,
            entries: self.entries,
            arena: self.arena,
            word_postings: HashMap::new(),
            trigram_postings: HashMap::new(),
        };
        idx.rebuild_postings();
        idx
    }
}

pub struct DeviceContextManager {
    storage_dir: PathBuf,
    config: RwLock<DeviceConfig>,
    index: RwLock<Arc<DeviceIndex>>,
    facts: RwLock<Vec<LearnedFact>>,
    indexing: AtomicBool,
}

static GLOBAL_DEVICE_MANAGER: OnceLock<DeviceContextManager> = OnceLock::new();

pub fn global_device_manager() -> &'static DeviceContextManager {
    GLOBAL_DEVICE_MANAGER.get_or_init(|| {
        let base = user_home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".kronos")
            .join("device_context");
        DeviceContextManager::new(base, true)
    })
}

impl DeviceContextManager {
    pub fn new(storage_dir: PathBuf, auto_load_system: bool) -> Self {
        let _ = fs::create_dir_all(&storage_dir);
        let cfg = load_config_file(&storage_dir.join("config.json"));
        let facts = load_facts_file(&storage_dir.join("learned_facts.json"));
        let k2di_path = storage_dir.join("device_index.k2di");
        let loaded_index = fs::read(&k2di_path)
            .ok()
            .and_then(|b| DeviceIndex::from_k2di_bytes(&b).ok())
            .unwrap_or_else(DeviceIndex::empty);

        let mgr = Self {
            storage_dir,
            config: RwLock::new(cfg.clone()),
            index: RwLock::new(Arc::new(loaded_index)),
            facts: RwLock::new(facts),
            indexing: AtomicBool::new(false),
        };
        if auto_load_system && cfg.enabled && mgr.current_index().is_empty() {
            mgr.rebuild_index_sync();
        }
        mgr
    }

    pub fn current_config(&self) -> DeviceConfig {
        self.config.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub fn current_index(&self) -> Arc<DeviceIndex> {
        Arc::clone(&self.index.read().unwrap_or_else(|p| p.into_inner()))
    }

    pub fn current_facts(&self) -> Vec<LearnedFact> {
        self.facts.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub fn status(&self) -> DeviceStatus {
        let cfg = self.current_config();
        let idx = self.current_index();
        let facts_len = self.facts.read().unwrap_or_else(|p| p.into_inner()).len();
        let (apps, docs, media, projs) = idx.counts_by_kind();
        DeviceStatus {
            enabled: cfg.enabled,
            allow_computer_control: cfg.allow_computer_control,
            indexing: self.indexing.load(Ordering::Relaxed),
            indexed_entries: idx.len(),
            apps_count: apps,
            docs_count: docs,
            media_count: media,
            projects_count: projs,
            learned_facts_count: facts_len,
            build_ms: idx.build_micros as f64 / 1000.0,
            artifact_bytes: idx.artifact_bytes,
            artifact_path: self.storage_dir.join("device_index.k2di").display().to_string(),
            profile_path: self.storage_dir.join("device_profile.md").display().to_string(),
            built_at: idx.built_at,
        }
    }

    pub fn status_json(&self) -> String {
        let s = self.status();
        format!(
            r#"{{"enabled":{},"allow_computer_control":{},"indexing":{},"indexed_entries":{},"apps_count":{},"docs_count":{},"media_count":{},"projects_count":{},"learned_facts_count":{},"build_ms":{:.2},"artifact_bytes":{},"artifact_path":{},"profile_path":{},"built_at":{}}}"#,
            s.enabled,
            s.allow_computer_control,
            s.indexing,
            s.indexed_entries,
            s.apps_count,
            s.docs_count,
            s.media_count,
            s.projects_count,
            s.learned_facts_count,
            s.build_ms,
            s.artifact_bytes,
            quote(&s.artifact_path),
            quote(&s.profile_path),
            s.built_at,
        )
    }

    pub fn update_config(
        &self,
        enabled: Option<bool>,
        allow_computer_control: Option<bool>,
        force_rebuild: bool,
    ) -> DeviceStatus {
        let mut need_build = force_rebuild;
        {
            let mut cfg = self.config.write().unwrap_or_else(|p| p.into_inner());
            if let Some(en) = enabled {
                if en && !cfg.enabled {
                    need_build = true;
                }
                cfg.enabled = en;
            }
            if let Some(cc) = allow_computer_control {
                cfg.allow_computer_control = cc;
            }
            cfg.updated_at = now_epoch_secs();
            let _ = save_config_file(&self.storage_dir.join("config.json"), &cfg);
        }
        if need_build || (self.current_config().enabled && self.current_index().is_empty()) {
            self.rebuild_index_sync();
        }
        self.status()
    }

    pub fn rebuild_index_sync(&self) {
        if self
            .indexing
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        let built = build_live_device_index();
        self.install_index(built);
        self.indexing.store(false, Ordering::SeqCst);
    }

    pub fn install_index(&self, mut built: DeviceIndex) {
        let bytes = built.to_k2di_bytes();
        built.artifact_bytes = bytes.len();
        let k2di_path = self.storage_dir.join("device_index.k2di");
        let tmp_path = self.storage_dir.join("device_index.k2di.tmp");
        if fs::write(&tmp_path, &bytes).is_ok() {
            let _ = fs::rename(&tmp_path, &k2di_path);
        }
        let arc = Arc::new(built);
        *self.index.write().unwrap_or_else(|p| p.into_inner()) = Arc::clone(&arc);
        let facts = self.current_facts();
        let profile_md = render_device_profile_markdown(&arc, &facts);
        let _ = fs::write(self.storage_dir.join("device_profile.md"), profile_md);
    }

    pub fn record_learned_fact(&self, key: &str, value: &str, path: &str) {
        let clean_key = key.trim();
        let clean_val = value.trim();
        if clean_key.is_empty() && clean_val.is_empty() {
            return;
        }
        let now = now_epoch_secs();
        let mut facts = self.facts.write().unwrap_or_else(|p| p.into_inner());
        if let Some(existing) = facts.iter_mut().find(|f| {
            (!path.is_empty() && f.path == path && f.key.eq_ignore_ascii_case(clean_key))
                || (path.is_empty() && f.key.eq_ignore_ascii_case(clean_key))
        }) {
            if !clean_val.is_empty() {
                existing.value = clean_val.to_string();
            }
            if !path.is_empty() {
                existing.path = path.to_string();
            }
            existing.updated_at = now;
            existing.hits = existing.hits.saturating_add(1);
        } else {
            facts.insert(
                0,
                LearnedFact {
                    key: clean_key.to_string(),
                    value: clean_val.to_string(),
                    path: path.to_string(),
                    updated_at: now,
                    hits: 1,
                },
            );
            if facts.len() > MAX_LEARNED_FACTS {
                facts.truncate(MAX_LEARNED_FACTS);
            }
        }
        let _ = save_facts_file(&self.storage_dir.join("learned_facts.json"), &facts);
        let snapshot_facts = facts.clone();
        drop(facts);
        let idx = self.current_index();
        if !idx.is_empty() {
            let md = render_device_profile_markdown(&idx, &snapshot_facts);
            let _ = fs::write(self.storage_dir.join("device_profile.md"), md);
        }
    }

    pub fn profile_markdown(&self) -> String {
        let path = self.storage_dir.join("device_profile.md");
        if let Ok(md) = fs::read_to_string(&path) {
            if !md.trim().is_empty() {
                return md;
            }
        }
        let idx = self.current_index();
        let facts = self.current_facts();
        render_device_profile_markdown(&idx, &facts)
    }

    /// Evaluates a user turn in sub-millisecond time and returns a compact context block only when relevant.
    pub fn query_for_turn(&self, user_prompt: &str) -> Option<DeviceTurnContext> {
        let trimmed = user_prompt.trim();
        if trimmed.is_empty() {
            return None;
        }
        let lower = trimmed.to_ascii_lowercase();
        let cfg = self.current_config();
        let wants_computer_action = cfg.allow_computer_control && is_computer_control_prompt(&lower);
        let wants_media = is_media_history_query(&lower);
        let wants_device_lookup = is_device_lookup_query(&lower);

        if !wants_computer_action && !wants_media && !wants_device_lookup {
            return None;
        }

        if !cfg.enabled && !wants_computer_action {
            return Some(DeviceTurnContext {
                prompt_block: "[Device Context: currently turned OFF in Kronos Settings. Tell the user they can enable 'Develop Device Context' in Settings or the Quick Prompt footer so Kronos indexes local documents, certificates, apps, and watched media history.]".to_string(),
                hits: Vec::new(),
                elapsed_ms: 0.01,
                wants_computer_action: false,
            });
        }

        // Ensure index is populated if enabled or if resolving an app/file to open.
        if self.current_index().is_empty() {
            self.rebuild_index_sync();
        }

        let t0 = Instant::now();
        let idx = self.current_index();
        let facts = self.current_facts();
        let mut lines = Vec::new();
        let mut collected_hits = Vec::new();

        if wants_media && cfg.enabled {
            let recent = idx.recent_watched_media(10);
            let movies: Vec<&DeviceHit> = recent.iter().filter(|h| !h.is_series_episode).collect();
            let episodes: Vec<&DeviceHit> = recent.iter().filter(|h| h.is_series_episode).collect();

            if !movies.is_empty() {
                lines.push("Recently watched movies on this Mac (ordered #1 = most recent):".to_string());
                for (rank, m) in movies.iter().take(5).enumerate() {
                    let disk_note = if m.exists_on_disk {
                        format!("path: {}", m.path)
                    } else {
                        "media file removed after playback".to_string()
                    };
                    lines.push(format!(
                        "  {}. {} — {} ({})",
                        rank + 1,
                        m.name,
                        m.detail,
                        disk_note
                    ));
                }
            }
            if !episodes.is_empty() {
                lines.push("Recently watched series episodes:".to_string());
                for ep in episodes.iter().take(3) {
                    lines.push(format!("  - {} — {}", ep.name, ep.detail));
                }
            }
            for h in recent.into_iter().take(6) {
                collected_hits.push(h);
            }
        }

        let kind_filter = if wants_computer_action && !wants_device_lookup && !wants_media {
            // Check both apps and files so "open whatsapp" and "open my ND certificate" both hit.
            None
        } else {
            None
        };
        let search_hits = idx.search(trimmed, &facts, kind_filter, 8);
        if !search_hits.is_empty() {
            lines.push("Matched local Mac items from Kronos Device Index:".to_string());
            for h in &search_hits {
                let status = if h.exists_on_disk { "on disk" } else { "history record" };
                let detail_part = if h.detail.is_empty() {
                    String::new()
                } else {
                    format!(" · {}", h.detail)
                };
                lines.push(format!(
                    "  - [{}] {} -> {} ({}{})",
                    h.kind.label(),
                    h.name,
                    h.path,
                    status,
                    detail_part
                ));
                if !collected_hits.iter().any(|c| c.path == h.path) {
                    collected_hits.push(h.clone());
                }
            }
        }

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if lines.is_empty() {
            if wants_computer_action {
                return Some(DeviceTurnContext {
                    prompt_block: format!(
                        "[Device Context · {:.2} ms]\nComputer control is active; call `use_computer` to open the requested application, file, or URL.",
                        elapsed_ms
                    ),
                    hits: Vec::new(),
                    elapsed_ms,
                    wants_computer_action: true,
                });
            }
            return None;
        }

        let action_hint = if wants_computer_action {
            "\nCall `use_computer` immediately with the exact path or application name above to carry out the action on macOS."
        } else {
            "\nAnswer directly and concisely using the exact local paths and details above."
        };
        let prompt_block = format!(
            "[Device Context · retrieved in {:.2} ms]\n{}{}",
            elapsed_ms,
            lines.join("\n"),
            action_hint
        );
        Some(DeviceTurnContext {
            prompt_block,
            hits: collected_hits,
            elapsed_ms,
            wants_computer_action,
        })
    }

    pub fn execute_computer_action(&self, args: &Json) -> Result<String, String> {
        let cfg = self.current_config();
        if !cfg.allow_computer_control {
            return Err("computer control is disabled in Kronos Device Context settings".to_string());
        }
        let action = args
            .get("action")
            .and_then(Json::as_str)
            .unwrap_or("")
            .trim();
        let target = args
            .get("target")
            .and_then(Json::as_str)
            .unwrap_or("")
            .trim();
        let note = args.get("note").and_then(Json::as_str).unwrap_or("").trim();

        match action {
            "open_app" => {
                if target.is_empty() {
                    return Err("`open_app` requires `target` application name".to_string());
                }
                if self.current_index().is_empty() {
                    self.rebuild_index_sync();
                }
                let idx = self.current_index();
                let facts = self.current_facts();
                let resolved_bundle = if target.ends_with(".app") && Path::new(target).exists() {
                    Some((
                        Path::new(target)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or(target)
                            .to_string(),
                        target.to_string(),
                    ))
                } else {
                    idx.search(target, &facts, Some(EntryKind::App), 3)
                        .into_iter()
                        .find(|h| h.exists_on_disk)
                        .map(|h| (h.name, h.path))
                };

                if let Some((app_name, bundle_path)) = resolved_bundle {
                    run_macos_open(&["-a", &bundle_path])?;
                    self.record_learned_fact(&format!("app:{target}"), &app_name, &bundle_path);
                    Ok(format!("opened application {app_name} ({bundle_path})"))
                } else {
                    run_macos_open(&["-a", target])?;
                    self.record_learned_fact(&format!("app:{target}"), target, target);
                    Ok(format!("opened application {target}"))
                }
            }
            "open_file" | "reveal_in_finder" => {
                if target.is_empty() {
                    return Err(format!("`{action}` requires `target` path or file query"));
                }
                let expanded = expand_tilde_path(target);
                let resolved = if expanded.exists() {
                    expanded
                } else {
                    if self.current_index().is_empty() {
                        self.rebuild_index_sync();
                    }
                    let idx = self.current_index();
                    let facts = self.current_facts();
                    let hit = idx
                        .search(target, &facts, None, 5)
                        .into_iter()
                        .find(|h| h.exists_on_disk && h.kind != EntryKind::App)
                        .ok_or_else(|| format!("no existing file matched `{target}`"))?;
                    PathBuf::from(hit.path)
                };
                let full_str = resolved.display().to_string();
                if action == "reveal_in_finder" {
                    run_macos_open(&["-R", &full_str])?;
                    self.record_learned_fact(target, "revealed in Finder", &full_str);
                    Ok(format!("revealed in Finder: {full_str}"))
                } else {
                    run_macos_open(&[&full_str])?;
                    self.record_learned_fact(target, "opened local file", &full_str);
                    Ok(format!("opened file: {full_str}"))
                }
            }
            "open_url" => {
                if !target.starts_with("http://") && !target.starts_with("https://") {
                    return Err("`open_url` requires an http:// or https:// URL".to_string());
                }
                run_macos_open(&[target])?;
                Ok(format!("opened URL: {target}"))
            }
            "search_device" => {
                if self.current_index().is_empty() {
                    self.rebuild_index_sync();
                }
                let t0 = Instant::now();
                let idx = self.current_index();
                let facts = self.current_facts();
                let hits = idx.search(target, &facts, None, 10);
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                if hits.is_empty() {
                    Ok(format!("0 matches in device index ({ms:.2} ms) for `{target}`"))
                } else {
                    let mut out = format!("{} matches ({ms:.2} ms):\n", hits.len());
                    for h in hits {
                        out.push_str(&format!(
                            "- [{}] {} -> {} ({})\n",
                            h.kind.label(),
                            h.name,
                            h.path,
                            if h.exists_on_disk { "on disk" } else { "history" }
                        ));
                    }
                    Ok(out)
                }
            }
            "recent_media" => {
                if self.current_index().is_empty() {
                    self.rebuild_index_sync();
                }
                let recent = self.current_index().recent_watched_media(10);
                if recent.is_empty() {
                    Ok("no recent watched media history recorded on this Mac".to_string())
                } else {
                    let mut out = String::from("Recent watched media (#1 = most recent):\n");
                    for (i, h) in recent.iter().enumerate() {
                        let kind_tag = if h.is_series_episode { "episode" } else { "movie" };
                        out.push_str(&format!(
                            "{}. [{}] {} — {} ({})\n",
                            i + 1,
                            kind_tag,
                            h.name,
                            h.detail,
                            if h.exists_on_disk { h.path.as_str() } else { "file removed after playback" }
                        ));
                    }
                    Ok(out)
                }
            }
            "system_info" => Ok(collect_macos_system_summary()),
            "set_volume" => {
                let vol: u32 = target
                    .trim_end_matches('%')
                    .parse()
                    .map_err(|_| "`set_volume` target must be an integer 0..100".to_string())?;
                let clamped = vol.min(100);
                let script = format!("set volume output volume {clamped}");
                let status = Command::new("/usr/bin/osascript")
                    .arg("-e")
                    .arg(&script)
                    .status()
                    .map_err(|e| format!("osascript failed: {e}"))?;
                if status.success() {
                    Ok(format!("system output volume set to {clamped}%"))
                } else {
                    Err("failed to set system volume".to_string())
                }
            }
            "remember_fact" => {
                let val = if note.is_empty() { target } else { note };
                self.record_learned_fact(target, val, "");
                Ok(format!("saved learned device fact: {target} -> {val}"))
            }
            other => Err(format!(
                "unknown `use_computer` action `{other}`; expected open_app, open_file, reveal_in_finder, open_url, search_device, recent_media, system_info, set_volume, or remember_fact"
            )),
        }
    }
}

pub fn use_computer_tool_definition_json() -> String {
    r#"{"type":"function","function":{"name":"use_computer","description":"Control and query the local macOS computer: open an installed application (such as WhatsApp, VLC, Safari, Slack), open a local file or certificate in its default macOS app, reveal a file in Finder, open a URL, search the sub-millisecond device index, check recent watched movies/media, get system status, set output volume, or save a learned device fact.","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["open_app","open_file","reveal_in_finder","open_url","search_device","recent_media","system_info","set_volume","remember_fact"],"description":"Computer action to execute on macOS"},"target":{"type":"string","description":"Application name (e.g. 'WhatsApp'), absolute file path, URL, search query, or volume 0-100"},"note":{"type":"string","description":"Optional value when saving a fact with remember_fact"}},"required":["action","target"]}}}"#.to_string()
}

pub fn is_computer_control_prompt(lower: &str) -> bool {
    let trimmed = lower.trim();
    // Avoid hijacking coding workspace prompts like "create index.html" or "edit style.css".
    let workspace_code_markers = [
        "write_file",
        "edit_file",
        ".rs",
        ".js",
        ".ts",
        ".css",
        "index.html",
        "style.css",
        "in workspace",
        "in the workspace",
    ];
    if workspace_code_markers.iter().any(|m| trimmed.contains(m)) {
        return false;
    }

    let direct_prefixes = [
        "open ",
        "launch ",
        "start ",
        "reveal ",
        "show in finder",
        "open up ",
        "can you open ",
        "please open ",
        "could you open ",
        "kronos open ",
        "set volume ",
        "mute volume",
    ];
    if direct_prefixes.iter().any(|p| trimmed.starts_with(p)) {
        return true;
    }
    if (trimmed.contains(" and open it")
        || trimmed.contains(" then open it")
        || trimmed.contains(", open it")
        || trimmed == "open it"
        || trimmed == "reveal it"
        || trimmed.contains("reveal it in finder"))
        && !trimmed.contains("html")
    {
        return true;
    }
    false
}

pub fn is_media_history_query(lower: &str) -> bool {
    let has_media_noun = [
        "movie", "movies", "film", "films", "watch", "watched", "watching", "episode", "series",
        "vlc", "video", "played",
    ]
    .iter()
    .any(|w| contains_word(lower, w));
    let has_query_cue = [
        "last", "recent", "recently", "what", "which", "history", "yesterday", "did",
    ]
    .iter()
    .any(|w| contains_word(lower, w));
    has_media_noun && (has_query_cue || lower.contains("watch"))
}

pub fn is_device_lookup_query(lower: &str) -> bool {
    let personal_doc_terms = [
        "certificate",
        "certificat",
        "certificates",
        "diploma",
        "transcript",
        "resume",
        "passport",
        "agreement",
        "contract",
        "timesheet",
        "invoice",
        "receipt",
        "downloads",
        "desktop",
        "documents",
        "applications",
        "installed",
    ];
    if personal_doc_terms.iter().any(|w| lower.contains(w)) {
        return true;
    }
    if (lower.starts_with("where is my ")
        || lower.starts_with("where's my ")
        || lower.starts_with("wheres my ")
        || lower.starts_with("find my ")
        || lower.starts_with("locate my ")
        || lower.contains("on my computer")
        || lower.contains("on my mac")
        || lower.contains("on this mac")
        || lower.contains("on my device"))
        && !lower.contains("index.html")
    {
        return true;
    }
    false
}

fn is_app_action_query(lower: &str) -> bool {
    is_computer_control_prompt(lower)
        && !lower.contains(".pdf")
        && !lower.contains(".docx")
        && !lower.contains("certificat")
        && !lower.contains("result")
        && !lower.contains("/users/")
}

fn contains_word(haystack_lower: &str, word: &str) -> bool {
    haystack_lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|w| w == word)
}

fn run_macos_open(args: &[&str]) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let out = Command::new("/usr/bin/open")
            .args(args)
            .output()
            .map_err(|e| format!("failed to run /usr/bin/open: {e}"))?;
        if out.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            Err(if stderr.is_empty() {
                format!("/usr/bin/open exited with {}", out.status)
            } else {
                stderr
            })
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = args;
        Err("macOS /usr/bin/open is only available on macOS".to_string())
    }
}

pub fn build_live_device_index() -> DeviceIndex {
    let t0 = Instant::now();
    let mut b = IndexBuilder::new();
    let home = user_home_dir().unwrap_or_else(|| PathBuf::from("/Users"));

    // 1. Installed macOS Applications.
    let mut app_dirs = vec![
        PathBuf::from("/Applications"),
        PathBuf::from("/Applications/Utilities"),
        PathBuf::from("/System/Applications"),
        PathBuf::from("/System/Applications/Utilities"),
        home.join("Applications"),
        home.join("Applications/Chrome Apps.localized"),
    ];
    let finder_path = PathBuf::from("/System/Library/CoreServices/Finder.app");
    if finder_path.exists() {
        b.add_entry(
            EntryKind::App,
            "Finder",
            "/System/Library/CoreServices/Finder.app",
            "macOS Finder file manager",
            FLAG_EXISTS_ON_DISK,
            0,
            0,
            0,
        );
    }
    for dir in app_dirs.drain(..) {
        scan_applications_dir(&dir, 0, &mut b);
    }

    // 2. Media watch history (VLC & IINA plists via native Foundation NSDictionary).
    scan_media_watch_history(&home, &mut b);

    // 3. Personal user folders (Downloads, Desktop, Movies, Pictures, Music).
    for folder in ["Downloads", "Desktop", "Movies", "Pictures", "Music"] {
        let root = home.join(folder);
        if root.is_dir() {
            scan_personal_dir(&root, folder, 0, 2, false, &mut b);
        }
    }

    // 4. Documents (recording git/code project roots cleanly without crawling 10k source files).
    let docs_root = home.join("Documents");
    if docs_root.is_dir() {
        scan_personal_dir(&docs_root, "Documents", 0, 3, true, &mut b);
    }

    let build_micros = t0.elapsed().as_micros() as u64;
    b.finish(build_micros)
}

fn scan_applications_dir(dir: &Path, depth: usize, b: &mut IndexBuilder) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if fname.starts_with('.') {
            continue;
        }
        if let Some(stem) = fname.strip_suffix(".app") {
            let mtime = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            b.add_entry(
                EntryKind::App,
                stem,
                &path.display().to_string(),
                "installed macOS application",
                FLAG_EXISTS_ON_DISK,
                0,
                mtime,
                0,
            );
        } else if depth == 0 && path.is_dir() {
            scan_applications_dir(&path, depth + 1, b);
        }
    }
}

fn scan_personal_dir(
    dir: &Path,
    section: &str,
    depth: usize,
    max_depth: usize,
    detect_code_repos: bool,
    b: &mut IndexBuilder,
) {
    const SKIP_DIRS: &[&str] = &[
        "node_modules",
        "target",
        "dist",
        "build",
        "Pods",
        "DerivedData",
        "__pycache__",
        "venv",
        "Library",
        "models",
    ];

    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if fname.starts_with('.') || SKIP_DIRS.contains(&fname) {
            continue;
        }
        let Ok(ft) = entry.file_type() else { continue };
        let meta = entry.metadata().ok();
        let mtime = meta
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let size = meta.as_ref().map_or(0, |m| m.len());

        if ft.is_dir() {
            if detect_code_repos && is_software_repo_root(&path) {
                b.add_entry(
                    EntryKind::Project,
                    fname,
                    &path.display().to_string(),
                    &format!("software project repository in {section}"),
                    FLAG_EXISTS_ON_DISK,
                    0,
                    mtime,
                    0,
                );
                continue;
            }
            if depth == 0 {
                let kind = if looks_like_media_title(fname) {
                    EntryKind::Media
                } else {
                    EntryKind::Directory
                };
                let detail = if kind == EntryKind::Media {
                    format!("media folder in {section}")
                } else {
                    format!("folder in {section}")
                };
                b.add_entry(
                    kind,
                    &clean_media_display_title(fname),
                    &path.display().to_string(),
                    &detail,
                    FLAG_EXISTS_ON_DISK,
                    0,
                    mtime,
                    0,
                );
            }
            if depth < max_depth {
                scan_personal_dir(&path, section, depth + 1, max_depth, detect_code_repos, b);
            }
        } else if ft.is_file() {
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let Some((kind, tag)) = classify_personal_file_ext(&ext) else {
                continue;
            };
            let display_name = if kind == EntryKind::Media {
                clean_media_display_title(fname)
            } else {
                fname.to_string()
            };
            let detail = format!("{tag} in {section} ({})", format_human_bytes(size));
            let mut flags = FLAG_EXISTS_ON_DISK;
            if kind == EntryKind::Media && is_series_episode_name(fname) {
                flags |= FLAG_SERIES_EPISODE;
            }
            b.add_entry(
                kind,
                &display_name,
                &path.display().to_string(),
                &detail,
                flags,
                0,
                mtime,
                size,
            );
        }
    }
}

fn is_software_repo_root(dir: &Path) -> bool {
    [
        ".git",
        "Cargo.toml",
        "package.json",
        "mix.exs",
        "pyproject.toml",
        "go.mod",
        "build.gradle",
    ]
    .iter()
    .any(|marker| dir.join(marker).exists())
}

fn classify_personal_file_ext(ext: &str) -> Option<(EntryKind, &'static str)> {
    match ext {
        "pdf" => Some((EntryKind::Document, "PDF document")),
        "docx" | "doc" | "pages" | "odt" | "rtf" => Some((EntryKind::Document, "text document")),
        "xlsx" | "xls" | "numbers" | "csv" => Some((EntryKind::Document, "spreadsheet")),
        "pptx" | "ppt" | "key" => Some((EntryKind::Document, "presentation")),
        "epub" => Some((EntryKind::Document, "ebook")),
        "txt" | "md" => Some((EntryKind::Document, "note")),
        "pem" | "cer" | "crt" | "p12" | "mobileprovision" => {
            Some((EntryKind::Document, "certificate file"))
        }
        "png" | "jpg" | "jpeg" | "heic" | "webp" => Some((EntryKind::Document, "image")),
        "zip" | "dmg" | "pkg" => Some((EntryKind::Document, "archive")),
        "mp4" | "mkv" | "mov" | "m4v" | "avi" | "webm" => Some((EntryKind::Media, "video")),
        "mp3" | "m4a" | "flac" | "wav" | "aac" => Some((EntryKind::Media, "audio")),
        _ => None,
    }
}

fn scan_media_watch_history(home: &Path, b: &mut IndexBuilder) {
    let vlc_plist = home.join("Library/Preferences/org.videolan.vlc.plist");
    if vlc_plist.exists() {
        for (rank_idx, (raw_url, pos_secs)) in read_vlc_recent_media(&vlc_plist).into_iter().enumerate() {
            let decoded_path = decode_file_url_to_path(&raw_url);
            if decoded_path.is_empty() {
                continue;
            }
            let p = Path::new(&decoded_path);
            let raw_fname = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(&decoded_path);
            let clean_title = clean_media_display_title(raw_fname);
            let file_exists = p.exists();
            let parent_folder = p.parent().filter(|par| {
                par.exists()
                    && *par != home.join("Downloads")
                    && *par != home.join("Movies")
                    && *par != home.join("Desktop")
            });
            let effective_path = if file_exists {
                decoded_path.clone()
            } else if let Some(par) = parent_folder {
                par.display().to_string()
            } else {
                decoded_path.clone()
            };
            let on_disk = file_exists || parent_folder.is_some();
            let mtime = fs::metadata(&effective_path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());

            let is_ep = is_series_episode_name(raw_fname);
            let mut flags = FLAG_WATCHED_HISTORY;
            if on_disk {
                flags |= FLAG_EXISTS_ON_DISK;
            }
            if is_ep {
                flags |= FLAG_SERIES_EPISODE;
            }
            let pos_note = if let Some(sec) = pos_secs {
                format!("watched in VLC (playback position {})", format_duration_hms(sec))
            } else {
                "watched in VLC".to_string()
            };
            let mru_rank = (rank_idx + 1).min(u16::MAX as usize) as u16;
            b.add_entry(
                EntryKind::Media,
                &clean_title,
                &effective_path,
                &pos_note,
                flags,
                mru_rank,
                mtime,
                0,
            );
        }
    }
}

/// Reads VLC's `recentlyPlayedMediaList` (ordered most-recent-first by reversing VLC's append queue)
/// and `recentlyPlayedMedia` playback timestamps directly via Foundation `NSDictionary` without spawning a subprocess.
#[cfg(target_os = "macos")]
fn read_vlc_recent_media(plist_path: &Path) -> Vec<(String, Option<u64>)> {
    use std::ffi::{c_char, c_void, CStr, CString};

    type Id = *mut c_void;
    type Sel = *mut c_void;

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
        fn objc_autoreleasePoolPush() -> *mut c_void;
        fn objc_autoreleasePoolPop(pool: *mut c_void);
    }

    #[link(name = "Foundation", kind = "framework")]
    unsafe extern "C" {}

    unsafe fn cls(name: &str) -> Id {
        let c = CString::new(name).unwrap();
        unsafe { objc_getClass(c.as_ptr()) }
    }
    unsafe fn sel(name: &str) -> Sel {
        let c = CString::new(name).unwrap();
        unsafe { sel_registerName(c.as_ptr()) }
    }
    unsafe fn ns_str(s: &str) -> Id {
        let Ok(c) = CString::new(s) else {
            return std::ptr::null_mut();
        };
        unsafe {
            let f: unsafe extern "C" fn(Id, Sel, *const c_char) -> Id =
                std::mem::transmute(objc_msgSend as *const ());
            f(cls("NSString"), sel("stringWithUTF8String:"), c.as_ptr())
        }
    }
    unsafe fn to_rust_str(ns: Id) -> Option<String> {
        if ns.is_null() {
            return None;
        }
        unsafe {
            let f: unsafe extern "C" fn(Id, Sel) -> *const c_char =
                std::mem::transmute(objc_msgSend as *const ());
            let ptr = f(ns, sel("UTF8String"));
            if ptr.is_null() {
                None
            } else {
                Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
            }
        }
    }

    let Some(path_str) = plist_path.to_str() else {
        return Vec::new();
    };

    unsafe {
        let pool = objc_autoreleasePoolPush();
        let mut out = Vec::new();
        let ns_p = ns_str(path_str);
        if !ns_p.is_null() {
            let dict_fn: unsafe extern "C" fn(Id, Sel, Id) -> Id =
                std::mem::transmute(objc_msgSend as *const ());
            let root = dict_fn(
                cls("NSDictionary"),
                sel("dictionaryWithContentsOfFile:"),
                ns_p,
            );
            if !root.is_null() {
                let get_fn: unsafe extern "C" fn(Id, Sel, Id) -> Id =
                    std::mem::transmute(objc_msgSend as *const ());
                let list = get_fn(root, sel("objectForKey:"), ns_str("recentlyPlayedMediaList"));
                let pos_dict = get_fn(root, sel("objectForKey:"), ns_str("recentlyPlayedMedia"));
                if !list.is_null() {
                    let count_fn: unsafe extern "C" fn(Id, Sel) -> usize =
                        std::mem::transmute(objc_msgSend as *const ());
                    let at_fn: unsafe extern "C" fn(Id, Sel, usize) -> Id =
                        std::mem::transmute(objc_msgSend as *const ());
                    let int_fn: unsafe extern "C" fn(Id, Sel) -> i64 =
                        std::mem::transmute(objc_msgSend as *const ());
                    let n = count_fn(list, sel("count"));
                    for i in (0..n).rev() {
                        let item = at_fn(list, sel("objectAtIndex:"), i);
                        if let Some(url) = to_rust_str(item) {
                            let secs = if !pos_dict.is_null() {
                                let num = get_fn(pos_dict, sel("objectForKey:"), item);
                                if !num.is_null() {
                                    let v = int_fn(num, sel("longLongValue"));
                                    (v > 0).then_some(v as u64)
                                } else {
                                    None
                                }
                            } else {
                                None
                            };
                            out.push((url, secs));
                        }
                    }
                }
            }
        }
        objc_autoreleasePoolPop(pool);
        out
    }
}

#[cfg(not(target_os = "macos"))]
fn read_vlc_recent_media(_plist_path: &Path) -> Vec<(String, Option<u64>)> {
    Vec::new()
}

pub fn decode_file_url_to_path(raw: &str) -> String {
    let stripped = raw
        .strip_prefix("file://localhost")
        .or_else(|| raw.strip_prefix("file://"))
        .unwrap_or(raw);
    percent_decode_utf8(stripped)
}

fn percent_decode_utf8(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push((h << 4) | l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

pub fn clean_media_display_title(raw_name: &str) -> String {
    let stem = raw_name
        .rsplit_once('.')
        .map_or(raw_name, |(base, ext)| {
            if matches!(
                ext.to_ascii_lowercase().as_str(),
                "mp4" | "mkv" | "mov" | "avi" | "m4v" | "webm"
            ) {
                base
            } else {
                raw_name
            }
        });
    let spaced: String = stem
        .chars()
        .map(|c| if c == '.' || c == '_' { ' ' } else { c })
        .collect();

    let noise_tags = [
        "1080p", "2160p", "720p", "480p", "bluray", "brrip", "webrip", "web-dl", "x264", "x265",
        "hevc", "10bit", "aac5", "aac", "eac3", "ac3", "ddp5", "yify", "yts",
    ];
    let mut kept = Vec::new();
    for raw_tok in spaced.split_whitespace() {
        let clean_low = raw_tok
            .trim_matches(|c: char| !c.is_ascii_alphanumeric())
            .to_ascii_lowercase();
        if noise_tags.iter().any(|t| clean_low == *t || clean_low.starts_with("x265-")) {
            break;
        }
        if raw_tok.starts_with('[') && clean_low != "1080p" {
            break;
        }
        kept.push(raw_tok);
    }
    if kept.is_empty() {
        return spaced.split_whitespace().collect::<Vec<_>>().join(" ");
    }
    // Wrap trailing 4-digit year in parentheses if not already parenthesized.
    if kept.len() >= 2 {
        let last = kept[kept.len() - 1];
        if last.len() == 4
            && last.chars().all(|c| c.is_ascii_digit())
            && (last.starts_with("19") || last.starts_with("20"))
        {
            let last_idx = kept.len() - 1;
            return format!("{} ({last})", kept[..last_idx].join(" "));
        }
    }
    kept.join(" ")
}

fn looks_like_media_title(folder_name: &str) -> bool {
    let low = folder_name.to_ascii_lowercase();
    ["1080p", "2160p", "720p", "bluray", "webrip", "web-dl", "season "]
        .iter()
        .any(|m| low.contains(m))
}

pub fn is_series_episode_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    for i in 0..bytes.len().saturating_sub(4) {
        if (bytes[i] == b'S' || bytes[i] == b's') && bytes[i + 1].is_ascii_digit() {
            let mut j = i + 2;
            if j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j + 1 < bytes.len()
                && (bytes[j] == b'E' || bytes[j] == b'e')
                && bytes[j + 1].is_ascii_digit()
            {
                return true;
            }
        }
    }
    false
}

fn tokenize_words_with_acronyms(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            tokens.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }

    // Add initial-letter acronym tokens (e.g. "Notre Dame Certificate" -> "ndc", "nd").
    let alpha_words: Vec<&str> = tokens
        .iter()
        .map(String::as_str)
        .filter(|w| w.chars().next().is_some_and(|c| c.is_ascii_alphabetic()))
        .collect();
    if alpha_words.len() >= 2 {
        let initials: String = alpha_words
            .iter()
            .take(6)
            .filter_map(|w| w.chars().next())
            .collect();
        if initials.len() >= 2 {
            if initials.len() > 2 {
                tokens.push(initials[..2].to_string());
            }
            tokens.push(initials);
        }
    }
    tokens
}

fn extract_meaningful_query_tokens(lower: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "the", "my", "me", "i", "you", "your", "is", "are", "was", "were", "did", "do",
        "does", "where", "wheres", "what", "whats", "which", "who", "how", "can", "could", "would",
        "please", "find", "locate", "show", "open", "launch", "start", "reveal", "in", "on", "at",
        "to", "for", "of", "it", "this", "that", "last", "recent", "recently", "computer", "mac",
        "device", "kronos", "and", "or", "file", "app", "application",
    ];
    let mut out = Vec::new();
    for raw in lower.split(|c: char| !c.is_ascii_alphanumeric()) {
        if raw.len() < 2 || STOPWORDS.contains(&raw) {
            continue;
        }
        let s = raw.to_string();
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

fn expand_query_synonyms(query_tokens: &[String]) -> Vec<&'static str> {
    let mut syns = Vec::new();
    for t in query_tokens {
        if t.starts_with("cert") || t.starts_with("diplom") || t.starts_with("degre") {
            for s in [
                "certificate",
                "result",
                "results",
                "statement",
                "diploma",
                "degree",
                "transcript",
            ] {
                if !syns.contains(&s) {
                    syns.push(s);
                }
            }
        } else if t == "cv" || t == "resume" {
            for s in ["resume", "cv", "curriculum"] {
                if !syns.contains(&s) {
                    syns.push(s);
                }
            }
        }
    }
    syns
}

fn extract_trigrams(word: &str) -> Vec<u32> {
    let b = word.as_bytes();
    if b.len() < 3 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(b.len() - 2);
    for w in b.windows(3) {
        let k = ((w[0] as u32) << 16) | ((w[1] as u32) << 8) | (w[2] as u32);
        if !out.contains(&k) {
            out.push(k);
        }
    }
    out
}

fn fnv1a_u32(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

fn render_device_profile_markdown(idx: &DeviceIndex, facts: &[LearnedFact]) -> String {
    let (apps, docs, media, projs) = idx.counts_by_kind();
    let mut md = String::with_capacity(4096);
    md.push_str("# Kronos Device Context Artifact\n\n");
    md.push_str(&format!("- System: {}\n", collect_macos_system_summary()));
    md.push_str(&format!(
        "- Binary Index: {} entries ({apps} apps, {docs} documents, {media} media items, {projs} projects) built in {:.2} ms ({} bytes)\n\n",
        idx.len(),
        idx.build_micros as f64 / 1000.0,
        idx.artifact_bytes
    ));

    if !facts.is_empty() {
        md.push_str("## Learned Device Facts\n");
        for f in facts.iter().take(20) {
            if f.path.is_empty() {
                md.push_str(&format!("- {} -> {} (used {}x)\n", f.key, f.value, f.hits));
            } else {
                md.push_str(&format!(
                    "- {} -> `{}` ({}, used {}x)\n",
                    f.key, f.path, f.value, f.hits
                ));
            }
        }
        md.push('\n');
    }

    let recent_media = idx.recent_watched_media(8);
    if !recent_media.is_empty() {
        md.push_str("## Recent Watched Media\n");
        for (i, m) in recent_media.iter().enumerate() {
            let tag = if m.is_series_episode { "series" } else { "movie" };
            md.push_str(&format!("{}. [{}] {} — {}\n", i + 1, tag, m.name, m.detail));
        }
        md.push('\n');
    }

    md.push_str("## Installed Applications (Sample)\n");
    let mut app_names: Vec<&str> = idx
        .entries
        .iter()
        .filter(|e| e.kind == EntryKind::App)
        .map(|e| idx.str_slice(e.name_off, e.name_len))
        .collect();
    app_names.sort_unstable();
    app_names.dedup();
    if !app_names.is_empty() {
        md.push_str(&app_names.into_iter().take(40).collect::<Vec<_>>().join(", "));
        md.push_str("\n\n");
    }

    let mut recent_docs: Vec<DeviceHit> = idx
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.kind == EntryKind::Document)
        .map(|(i, _)| idx.entry_to_hit(i, 0.0))
        .collect();
    recent_docs.sort_by(|a, b| b.modified_epoch.cmp(&a.modified_epoch));
    if !recent_docs.is_empty() {
        md.push_str("## Recent Documents & Certificates\n");
        for d in recent_docs.into_iter().take(15) {
            md.push_str(&format!("- `{}` ({})\n", d.path, d.detail));
        }
    }
    md
}

fn collect_macos_system_summary() -> String {
    let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
    let home = user_home_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "~".to_string());
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    format!("user={user}, home={home}, os={os}, arch={arch}")
}

fn user_home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn expand_tilde_path(raw: &str) -> PathBuf {
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(h) = user_home_dir() {
            return h.join(rest);
        }
    }
    PathBuf::from(raw)
}

fn format_human_bytes(bytes: u64) -> String {
    if bytes >= (1 << 30) {
        format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= (1 << 20) {
        format!("{:.1} MiB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= (1 << 10) {
        format!("{:.0} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn format_duration_hms(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h {m:02}m {s:02}s")
    } else {
        format!("{m}m {s:02}s")
    }
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn load_config_file(path: &Path) -> DeviceConfig {
    let Ok(txt) = fs::read_to_string(path) else {
        return DeviceConfig::default();
    };
    let Ok(j) = Json::parse(&txt) else {
        return DeviceConfig::default();
    };
    DeviceConfig {
        enabled: j.get("enabled").and_then(Json::as_bool).unwrap_or(false),
        allow_computer_control: j
            .get("allow_computer_control")
            .and_then(Json::as_bool)
            .unwrap_or(true),
        updated_at: j
            .get("updated_at")
            .and_then(Json::as_f64)
            .map_or_else(now_epoch_secs, |v| v as u64),
    }
}

fn save_config_file(path: &Path, cfg: &DeviceConfig) -> Result<(), String> {
    let json = format!(
        "{{\n  \"enabled\": {},\n  \"allow_computer_control\": {},\n  \"updated_at\": {}\n}}\n",
        cfg.enabled, cfg.allow_computer_control, cfg.updated_at
    );
    fs::write(path, json).map_err(|e| e.to_string())
}

fn load_facts_file(path: &Path) -> Vec<LearnedFact> {
    let Ok(txt) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(j) = Json::parse(&txt) else {
        return Vec::new();
    };
    let Some(arr) = j.as_arr() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let key = item.get("key").and_then(Json::as_str).unwrap_or("").to_string();
        let value = item.get("value").and_then(Json::as_str).unwrap_or("").to_string();
        let path_s = item.get("path").and_then(Json::as_str).unwrap_or("").to_string();
        if key.is_empty() && value.is_empty() {
            continue;
        }
        out.push(LearnedFact {
            key,
            value,
            path: path_s,
            updated_at: item.get("updated_at").and_then(Json::as_f64).unwrap_or(0.0) as u64,
            hits: item.get("hits").and_then(Json::as_f64).unwrap_or(1.0) as u32,
        });
    }
    out
}

fn save_facts_file(path: &Path, facts: &[LearnedFact]) -> Result<(), String> {
    let items: Vec<String> = facts
        .iter()
        .map(|f| {
            format!(
                r#"  {{"key":{},"value":{},"path":{},"updated_at":{},"hits":{}}}"#,
                quote(&f.key),
                quote(&f.value),
                quote(&f.path),
                f.updated_at,
                f.hits
            )
        })
        .collect();
    let json = format!("[\n{}\n]\n", items.join(",\n"));
    fs::write(path, json).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k2di_roundtrip_and_fuzzy_certificate_and_media_search() {
        let mut b = IndexBuilder::new();
        b.add_entry(
            EntryKind::App,
            "WhatsApp",
            "/Applications/WhatsApp.app",
            "installed macOS application",
            FLAG_EXISTS_ON_DISK,
            0,
            1_790_000_000,
            0,
        );
        b.add_entry(
            EntryKind::Document,
            "ND statement of result.pdf",
            "/Users/test/Downloads/ND statement of result.pdf",
            "PDF document in Downloads (420 KiB)",
            FLAG_EXISTS_ON_DISK,
            0,
            1_792_000_000,
            430_000,
        );
        b.add_entry(
            EntryKind::Document,
            "Secondary School Certificate.pdf",
            "/Users/test/Downloads/Secondary School Certificate.pdf",
            "PDF document in Downloads (550 KiB)",
            FLAG_EXISTS_ON_DISK,
            0,
            1_791_000_000,
            560_000,
        );
        b.add_entry(
            EntryKind::Document,
            "handimann_client_flow.html",
            "/Users/test/Downloads/handimann_client_flow.html",
            "note in Downloads (18 KiB)",
            FLAG_EXISTS_ON_DISK,
            0,
            1_792_500_000,
            18_000,
        );
        b.add_entry(
            EntryKind::Media,
            "The Invite (2026)",
            "/Users/test/Downloads/The Invite (2026)",
            "watched in VLC (playback position 1h 32m 45s)",
            FLAG_EXISTS_ON_DISK | FLAG_WATCHED_HISTORY,
            1,
            1_793_151_490,
            0,
        );
        b.add_entry(
            EntryKind::Media,
            "Andor S01E08",
            "/Users/test/Downloads/Andor S01E08.mp4",
            "watched in VLC (playback position 3m 05s)",
            FLAG_WATCHED_HISTORY | FLAG_SERIES_EPISODE,
            2,
            1_793_000_000,
            0,
        );

        let idx = b.finish(420);
        let bytes = idx.to_k2di_bytes();
        let loaded = DeviceIndex::from_k2di_bytes(&bytes).expect("k2di decode");
        assert_eq!(loaded.len(), 6);

        let cert_hits = loaded.search("where's my ND certificat", &[], None, 5);
        assert!(!cert_hits.is_empty());
        assert_eq!(
            cert_hits[0].path,
            "/Users/test/Downloads/ND statement of result.pdf"
        );

        let app_hits = loaded.search("open whatsapp", &[], Some(EntryKind::App), 3);
        assert_eq!(app_hits[0].path, "/Applications/WhatsApp.app");

        let movie_hits = loaded.search("what movie did I watch last", &[], None, 3);
        assert_eq!(movie_hits[0].name, "The Invite (2026)");
    }

    #[test]
    fn test_media_title_cleaning_and_episode_detection() {
        assert_eq!(
            clean_media_display_title("The.Invite.2026.1080p.WEBRip.x264.AAC5.1-[YTS.GG].mp4"),
            "The Invite (2026)"
        );
        assert!(is_series_episode_name("Andor S01E08.mp4"));
        assert!(!is_series_episode_name("The Social Network (2010) 1080p.mp4"));
        assert_eq!(
            decode_file_url_to_path("file:///Users/test/ND%20statement.pdf"),
            "/Users/test/ND statement.pdf"
        );
    }

    #[test]
    fn test_manager_persistence_and_selective_turn_gating() {
        let dir = std::env::temp_dir().join(format!(
            "kronos_dev_ctx_test_{}_{}",
            std::process::id(),
            now_epoch_secs()
        ));
        let _ = fs::remove_dir_all(&dir);
        let mgr = DeviceContextManager::new(dir.clone(), false);

        // Unrelated coding prompt incurs zero context injection.
        assert!(mgr.query_for_turn("implement merge sort in rust").is_none());

        let mut b = IndexBuilder::new();
        b.add_entry(
            EntryKind::Document,
            "ND statement of result.pdf",
            "/Users/test/Downloads/ND statement of result.pdf",
            "PDF document in Downloads",
            FLAG_EXISTS_ON_DISK,
            0,
            100,
            1024,
        );
        mgr.update_config(Some(true), Some(true), false);
        mgr.install_index(b.finish(150));
        mgr.record_learned_fact(
            "nd certificate",
            "National Diploma result PDF",
            "/Users/test/Downloads/ND statement of result.pdf",
        );

        let ctx = mgr
            .query_for_turn("where's my ND certificat")
            .expect("expected device context hit");
        assert!(ctx.prompt_block.contains("ND statement of result.pdf"));
        assert!(ctx.elapsed_ms < 25.0);

        let _ = fs::remove_dir_all(&dir);
    }
}
