//! Per-session KV prefixes: the last N conversations stay in RAM, the rest spill to `state_dir/kv`.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::model::kv_blob::KvBlobLayout;

const MAGIC: &[u8; 4] = b"KRKV";
const VERSION: u32 = 1;

/// Token-major KV bytes for `tokens`, plus how many of them are already on disk.
pub struct KvBlob {
    pub layout: KvBlobLayout,
    pub tokens: Vec<usize>,
    pub data: Vec<u8>,
    disk_len: usize,
}

impl KvBlob {
    pub fn new(layout: KvBlobLayout) -> Self {
        Self { layout, tokens: Vec::new(), data: Vec::new(), disk_len: 0 }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Keeps the first `n` tokens.
    pub fn truncate(&mut self, n: usize) {
        if n < self.tokens.len() {
            self.tokens.truncate(n);
            self.data.truncate(n * self.layout.bytes_per_pos as usize);
            self.disk_len = self.disk_len.min(n);
        }
    }

    /// Length of the shared prefix with `tokens`.
    pub fn common_prefix(&self, tokens: &[usize]) -> usize {
        self.tokens.iter().zip(tokens).take_while(|(a, b)| a == b).count()
    }

    /// Appends `tokens` whose KV bytes `export` writes into `self.data`.
    pub fn extend(&mut self, tokens: &[usize], export: impl FnOnce(&mut Vec<u8>) -> bool) -> bool {
        let before = self.data.len();
        if !export(&mut self.data) {
            self.data.truncate(before);
            return false;
        }
        debug_assert_eq!(self.data.len() - before, tokens.len() * self.layout.bytes_per_pos as usize);
        self.tokens.extend_from_slice(tokens);
        true
    }

    fn header(&self, tag: &str) -> Vec<u8> {
        let mut h = Vec::with_capacity(64 + tag.len());
        h.extend_from_slice(MAGIC);
        h.extend_from_slice(&VERSION.to_le_bytes());
        h.extend_from_slice(&(tag.len() as u32).to_le_bytes());
        h.extend_from_slice(tag.as_bytes());
        h.extend_from_slice(&self.layout.encode());
        h.extend_from_slice(&(self.tokens.len() as u64).to_le_bytes());
        h
    }

    /// Appends the tokens not yet on disk (`[header][data][tokens]`, header count written last).
    fn persist(&mut self, path: &Path, tag: &str) -> io::Result<()> {
        let bpp = self.layout.bytes_per_pos as usize;
        let header = self.header(tag);
        let n = self.tokens.len();
        if self.disk_len == 0 || !path.exists() {
            let tmp = path.with_extension("kv.tmp");
            let mut f = File::create(&tmp)?;
            f.write_all(&header)?;
            f.write_all(&self.data)?;
            f.write_all(&encode_tokens(&self.tokens))?;
            f.sync_data()?;
            fs::rename(&tmp, path)?;
        } else {
            let mut f = OpenOptions::new().write(true).open(path)?;
            f.set_len((header.len() + n * bpp) as u64)?;
            f.seek(SeekFrom::Start((header.len() + self.disk_len * bpp) as u64))?;
            f.write_all(&self.data[self.disk_len * bpp..])?;
            f.write_all(&encode_tokens(&self.tokens))?;
            f.seek(SeekFrom::Start(0))?;
            f.write_all(&header)?;
            f.sync_data()?;
        }
        self.disk_len = n;
        Ok(())
    }

    /// Loads a blob written by [`Self::persist`] when its tag and layout match.
    fn load(path: &Path, tag: &str, layout: &KvBlobLayout) -> Option<Self> {
        let mut f = File::open(path).ok()?;
        let mut fixed = [0u8; 12];
        f.read_exact(&mut fixed).ok()?;
        if &fixed[..4] != MAGIC || u32::from_le_bytes(fixed[4..8].try_into().ok()?) != VERSION {
            return None;
        }
        let tag_len = u32::from_le_bytes(fixed[8..12].try_into().ok()?) as usize;
        let mut rest = vec![0u8; tag_len + KvBlobLayout::ENCODED_LEN + 8];
        f.read_exact(&mut rest).ok()?;
        if &rest[..tag_len] != tag.as_bytes() {
            return None;
        }
        let file_layout = KvBlobLayout::decode(&rest[tag_len..])?;
        if file_layout != *layout {
            return None;
        }
        let n = u64::from_le_bytes(rest[tag_len + KvBlobLayout::ENCODED_LEN..].try_into().ok()?) as usize;
        let bpp = layout.bytes_per_pos as usize;
        let header_len = 12 + rest.len();
        let need = header_len + n * bpp + n * 4;
        if f.metadata().ok()?.len() < need as u64 {
            return None;
        }
        let mut data = vec![0u8; n * bpp];
        f.read_exact(&mut data).ok()?;
        let mut tok_bytes = vec![0u8; n * 4];
        f.read_exact(&mut tok_bytes).ok()?;
        let tokens = tok_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        Some(Self { layout: *layout, tokens, data, disk_len: n })
    }
}

fn encode_tokens(tokens: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 4);
    for &t in tokens {
        out.extend_from_slice(&(t as u32).to_le_bytes());
    }
    out
}

/// One conversation's target (and optional draft) KV prefix.
pub struct SessionKv {
    pub id: String,
    pub target: KvBlob,
    pub draft: Option<KvBlob>,
    last_used: u64,
}

impl SessionKv {
    pub fn bytes(&self) -> usize {
        self.target.data.len() + self.draft.as_ref().map_or(0, |d| d.data.len())
    }
}

/// LRU of at most `capacity` sessions in RAM; evicted or switched-away sessions persist under `dir`.
pub struct SessionKvStore {
    entries: Vec<SessionKv>,
    capacity: usize,
    dir: Option<PathBuf>,
    tag: String,
    tick: u64,
}

/// Where a restored prefix came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvSource {
    Active,
    Memory,
    Disk,
    None,
}

impl KvSource {
    pub fn as_str(self) -> &'static str {
        match self {
            KvSource::Active => "active",
            KvSource::Memory => "memory",
            KvSource::Disk => "disk",
            KvSource::None => "none",
        }
    }
}

impl SessionKvStore {
    /// `tag` names the model and context so files from another model are ignored.
    pub fn new(capacity: usize, dir: Option<PathBuf>, tag: String) -> Self {
        if let Some(d) = &dir {
            let _ = fs::create_dir_all(d);
        }
        Self { entries: Vec::new(), capacity: capacity.max(1), dir, tag, tick: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn resident_bytes(&self) -> usize {
        self.entries.iter().map(SessionKv::bytes).sum()
    }

    pub fn resident_ids(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.id.as_str()).collect()
    }

    fn paths(&self, id: &str) -> Option<(PathBuf, PathBuf)> {
        let d = self.dir.as_ref()?;
        Some((d.join(format!("{id}.kv")), d.join(format!("{id}.draft.kv"))))
    }

    /// Removes `id` from RAM.
    pub fn take(&mut self, id: &str) -> Option<SessionKv> {
        let i = self.entries.iter().position(|e| e.id == id)?;
        Some(self.entries.remove(i))
    }

    /// Reads `id` back from disk (`None` when absent, stale, or from a different layout).
    pub fn load(&self, id: &str, target: &KvBlobLayout, draft: Option<&KvBlobLayout>) -> Option<SessionKv> {
        let (tp, dp) = self.paths(id)?;
        let target = KvBlob::load(&tp, &self.tag, target)?;
        let draft = draft.and_then(|dl| KvBlob::load(&dp, &self.tag, dl));
        Some(SessionKv { id: id.to_string(), target, draft, last_used: 0 })
    }

    /// Inserts (or replaces) `entry` as most recently used.
    pub fn insert(&mut self, mut entry: SessionKv) -> &mut SessionKv {
        self.tick += 1;
        entry.last_used = self.tick;
        self.entries.retain(|e| e.id != entry.id);
        self.entries.push(entry);
        self.entries.last_mut().unwrap()
    }

    /// Returns the RAM entry for `id`, creating an empty one, and marks it most recently used.
    pub fn upsert(&mut self, id: &str, target: KvBlobLayout) -> &mut SessionKv {
        self.tick += 1;
        let tick = self.tick;
        if let Some(i) = self.entries.iter().position(|e| e.id == id) {
            self.entries[i].last_used = tick;
            return &mut self.entries[i];
        }
        self.entries.push(SessionKv { id: id.to_string(), target: KvBlob::new(target), draft: None, last_used: tick });
        self.entries.last_mut().unwrap()
    }

    /// Writes the not-yet-persisted tail of `entry` to disk.
    pub fn persist(&self, entry: &mut SessionKv) -> io::Result<()> {
        let Some((tp, dp)) = self.paths(&entry.id) else { return Ok(()) };
        entry.target.persist(&tp, &self.tag)?;
        if let Some(d) = entry.draft.as_mut() {
            d.persist(&dp, &self.tag)?;
        }
        Ok(())
    }

    /// [`Self::persist`] for the RAM entry `id`.
    pub fn persist_id(&mut self, id: &str) -> io::Result<()> {
        let Some((tp, dp)) = self.paths(id) else { return Ok(()) };
        let tag = self.tag.clone();
        let Some(e) = self.entries.iter_mut().find(|e| e.id == id) else { return Ok(()) };
        e.target.persist(&tp, &tag)?;
        if let Some(d) = e.draft.as_mut() {
            d.persist(&dp, &tag)?;
        }
        Ok(())
    }

    /// Spills least recently used sessions (never `keep`) until at most `capacity` remain; returns their ids.
    pub fn evict_to_capacity(&mut self, keep: &str) -> Vec<String> {
        let mut evicted = Vec::new();
        while self.entries.len() > self.capacity {
            let Some(i) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.id != keep)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
            else {
                break;
            };
            let mut e = self.entries.remove(i);
            if let Err(err) = self.persist(&mut e) {
                eprintln!("kv spill {}: {err}", e.id);
            }
            evicted.push(e.id);
        }
        evicted
    }

    /// Deletes `id` from RAM and disk.
    pub fn remove(&mut self, id: &str) {
        self.entries.retain(|e| e.id != id);
        if let Some((tp, dp)) = self.paths(id) {
            let _ = fs::remove_file(tp);
            let _ = fs::remove_file(dp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(bpp: u32) -> KvBlobLayout {
        KvBlobLayout { backend: 0, bits: 8, layers: 2, kv_heads: 1, head_dim: 64, bytes_per_pos: bpp }
    }

    #[test]
    fn test_kv_blob_persist_append_and_reload() {
        let dir = std::env::temp_dir().join(format!("kronos_kv_sessions_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = SessionKvStore::new(2, Some(dir.clone()), "m:1".into());
        let mut e = SessionKv { id: "s1".into(), target: KvBlob::new(layout(3)), draft: None, last_used: 0 };
        assert!(e.target.extend(&[5, 6], |out| {
            out.extend_from_slice(&[1, 1, 1, 2, 2, 2]);
            true
        }));
        store.persist(&mut e).unwrap();
        assert!(e.target.extend(&[7], |out| {
            out.extend_from_slice(&[3, 3, 3]);
            true
        }));
        store.persist(&mut e).unwrap();
        let back = store.load("s1", &layout(3), None).unwrap();
        assert_eq!(back.target.tokens, vec![5, 6, 7]);
        assert_eq!(back.target.data, vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        // Diverged prefix: truncate, append, reload.
        e.target.truncate(1);
        assert!(e.target.extend(&[9], |out| {
            out.extend_from_slice(&[4, 4, 4]);
            true
        }));
        store.persist(&mut e).unwrap();
        let back = store.load("s1", &layout(3), None).unwrap();
        assert_eq!(back.target.tokens, vec![5, 9]);
        assert_eq!(back.target.data, vec![1, 1, 1, 4, 4, 4]);
        assert!(store.load("s1", &layout(4), None).is_none());
        assert!(SessionKvStore::new(1, Some(dir.clone()), "other".into()).load("s1", &layout(3), None).is_none());

        let mut store = store;
        store.insert(e);
        store.upsert("s2", layout(3));
        store.upsert("s3", layout(3));
        let ev = store.evict_to_capacity("s3");
        assert_eq!(ev, vec!["s1".to_string()]);
        assert_eq!(store.resident_ids(), vec!["s2", "s3"]);
        let _ = fs::remove_dir_all(&dir);
    }
}
