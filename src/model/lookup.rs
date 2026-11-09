//! Session-wide prompt-lookup drafting: an n-gram index over the live context plus any other
//! token documents of the session (files the agent read, earlier turns), so drafts can copy text
//! the model is likely to repeat. Drafts are verified by the target model, so a bad draft costs
//! only verify time, never correctness.

use std::collections::HashMap;

/// Tokens that key the index.
const KEY: usize = 3;
/// Candidates scored per lookup (most recent first).
const MAX_CANDIDATES: usize = 64;
/// Longest backward match considered when ranking candidates.
const MAX_MATCH: usize = 16;

/// Token documents with an index from each `KEY`-gram to the positions where it ends.
#[derive(Default)]
pub struct SessionLookup {
    docs: Vec<Vec<usize>>,
    index: HashMap<[usize; KEY], Vec<(u32, u32)>>,
}

impl SessionLookup {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a document and returns its id.
    pub fn add_doc(&mut self, tokens: &[usize]) -> usize {
        let id = self.docs.len();
        self.docs.push(Vec::with_capacity(tokens.len()));
        for &t in tokens {
            self.push(id, t);
        }
        id
    }

    /// Appends `tok` to document `doc` and indexes the gram it completes.
    pub fn push(&mut self, doc: usize, tok: usize) {
        let d = &mut self.docs[doc];
        d.push(tok);
        let n = d.len();
        if n >= KEY {
            let key: [usize; KEY] = d[n - KEY..].try_into().expect("KEY tokens");
            self.index.entry(key).or_default().push((doc as u32, (n - 1) as u32));
        }
    }

    pub fn num_docs(&self) -> usize {
        self.docs.len()
    }

    pub fn doc(&self, doc: usize) -> &[usize] {
        &self.docs[doc]
    }

    /// Up to `k` tokens that followed the best earlier match of the end of document `ctx`, or an
    /// empty draft. The best match is the one with the longest exact backward overlap.
    pub fn draft(&self, ctx: usize, k: usize) -> Vec<usize> {
        self.draft_with(ctx, &[], k)
    }

    /// [`Self::draft`] as if `tail` were appended to document `ctx` (without indexing it).
    pub fn draft_with(&self, ctx: usize, tail: &[usize], k: usize) -> Vec<usize> {
        let d0 = &self.docs[ctx];
        let keep = MAX_MATCH.min(d0.len());
        let c: Vec<usize> = d0[d0.len() - keep..].iter().chain(tail).copied().collect();
        let c_full_len = d0.len() + tail.len();
        if c.len() < KEY || k == 0 {
            return Vec::new();
        }
        let key: [usize; KEY] = c[c.len() - KEY..].try_into().expect("KEY tokens");
        let Some(hits) = self.index.get(&key) else { return Vec::new() };
        let mut best: Option<(usize, usize, usize)> = None;
        for &(doc, end) in hits.iter().rev().take(MAX_CANDIDATES) {
            let (doc, end) = (doc as usize, end as usize);
            if doc == ctx && end + 1 >= c_full_len {
                continue;
            }
            let d = &self.docs[doc];
            if end + 1 >= d.len() {
                continue;
            }
            let mut m = KEY;
            while m < MAX_MATCH && m <= end && m < c.len() && d[end - m] == c[c.len() - 1 - m] {
                m += 1;
            }
            if best.is_none_or(|b| m > b.2) {
                best = Some((doc, end, m));
            }
        }
        let Some((doc, end, _)) = best else { return Vec::new() };
        let d = &self.docs[doc];
        d[end + 1..(end + 1 + k).min(d.len())].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafts_continue_the_longest_match() {
        let mut s = SessionLookup::new();
        // Two earlier occurrences of [7, 8, 9]; only the second shares a longer prefix with the context.
        s.add_doc(&[1, 7, 8, 9, 100, 101, 5, 6, 7, 8, 9, 200, 201, 202]);
        let ctx = s.add_doc(&[0, 5, 6, 7, 8]);
        s.push(ctx, 9);
        assert_eq!(s.draft(ctx, 3), vec![200, 201, 202]);
    }

    #[test]
    fn virtual_tail_matches_like_pushed_tokens() {
        let mut s = SessionLookup::new();
        let ctx = s.add_doc(&[1, 2, 3, 4, 5, 9, 1, 2]);
        assert_eq!(s.draft_with(ctx, &[3], 3), vec![4, 5, 9]);
    }

    #[test]
    fn context_never_drafts_from_its_own_end() {
        let mut s = SessionLookup::new();
        let ctx = s.add_doc(&[4, 5, 6]);
        assert!(s.draft(ctx, 4).is_empty());
        for t in [9, 4, 5, 6] {
            s.push(ctx, t);
        }
        assert_eq!(s.draft(ctx, 4), vec![9, 4, 5, 6]);
    }
}
