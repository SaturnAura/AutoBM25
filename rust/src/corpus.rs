//! Single-pass corpus preparation.
//!
//! The Python implementation walks the corpus twice with a full Python loop: once
//! in `extract_features` (keeping a per-document dict for *every* document in
//! memory) and once in `BM25Engine.build_index`. Here one pass produces both the
//! inverted index and the per-document statistics the features are computed from,
//! so tokenization happens once and nothing per-document is retained beyond the
//! small numeric vectors.
//!
//! Terms also live in a flat table (`postings` / `vocab_freq` indexed by term id),
//! so the hot loop hashes once per token occurrence and never re-hashes per posting.

use std::collections::HashSet;

use crate::fx::FxHashMap;
use crate::loader::Doc;
use crate::text::TokenBuffer;

/// Per-term bookkeeping during the build; one entry per distinct term.
#[derive(Clone, Copy)]
struct TermSlot {
    /// marker of the last document this term appeared in (0 = never)
    marker: u32,
    /// term frequency inside that document
    count: u32,
    /// id of this term (index into `postings` / `vocab_freq`)
    term_id: u32,
    /// position of this term inside the current document's term list
    doc_pos: u32,
}

pub struct Corpus {
    /// doc_idx -> doc_id
    pub doc_ids: Vec<String>,
    /// doc_idx -> token count
    pub doc_len: Vec<u32>,
    /// term -> term id
    pub term_index: FxHashMap<String, u32>,
    /// term id -> postings, (doc_idx, tf) in ascending doc order
    pub postings: Vec<Vec<(u32, u32)>>,
    /// term id -> total term frequency across the corpus
    pub vocab_freq: Vec<u64>,
    /// per-document unique/total ratio (0.0 for empty documents)
    pub ttrs: Vec<f64>,
    /// per-document max tf over non-stopword terms
    pub max_tfs: Vec<f64>,
    /// per-document mean tf = tokens / unique terms
    pub mean_tfs: Vec<f64>,
    /// Heaps' law points: (cumulative tokens, cumulative vocabulary)
    pub heaps_points: Vec<(u64, u64)>,
    pub stopword_tokens: u64,
    pub total_tokens: u64,
    pub avgdl: f64,
}

impl Corpus {
    pub fn build(docs: &[Doc], stopwords: &HashSet<String>) -> Corpus {
        let n = docs.len();
        let mut doc_ids: Vec<String> = Vec::with_capacity(n);
        let mut doc_len: Vec<u32> = vec![0u32; n];
        let mut ttrs: Vec<f64> = Vec::with_capacity(n);
        let mut max_tfs: Vec<f64> = Vec::with_capacity(n);
        let mut mean_tfs: Vec<f64> = Vec::with_capacity(n);
        let mut heaps_points: Vec<(u64, u64)> = Vec::with_capacity(n);

        let mut counts: FxHashMap<String, TermSlot> = FxHashMap::default();
        let mut postings: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut vocab_freq: Vec<u64> = Vec::new();
        let mut is_stopword: Vec<bool> = Vec::new();

        let mut doc_terms: Vec<u32> = Vec::new();
        let mut doc_counts: Vec<u32> = Vec::new();
        let mut tb = TokenBuffer::new();
        let mut vocab_seen: u64 = 0;
        let mut cum_tokens: u64 = 0;
        let mut total_tokens: u64 = 0;
        let mut stopword_tokens: u64 = 0;

        for (i, d) in docs.iter().enumerate() {
            doc_ids.push(d.id.clone());
            tb.refill(&d.text);
            let ntok = tb.len();
            doc_len[i] = ntok as u32;
            total_tokens += ntok as u64;
            cum_tokens += ntok as u64;

            let marker = i as u32 + 1;
            doc_terms.clear();
            doc_counts.clear();
            let mut new_terms: u64 = 0;

            for tok in tb.tokens() {
                let mut first_in_doc: Option<u32> = None;
                if let Some(slot) = counts.get_mut(tok) {
                    if slot.marker == marker {
                        slot.count += 1;
                        let pos = slot.doc_pos as usize;
                        doc_counts[pos] += 1;
                    } else {
                        slot.marker = marker;
                        slot.count = 1;
                        slot.doc_pos = doc_terms.len() as u32;
                        first_in_doc = Some(slot.term_id);
                    }
                } else {
                    let term_id = postings.len() as u32;
                    let doc_pos = doc_terms.len() as u32;
                    postings.push(Vec::new());
                    vocab_freq.push(0);
                    is_stopword.push(stopwords.contains(tok));
                    counts.insert(
                        tok.to_string(),
                        TermSlot { marker, count: 1, term_id, doc_pos },
                    );
                    doc_terms.push(term_id);
                    doc_counts.push(1);
                    new_terms += 1;
                    continue;
                }
                if let Some(term_id) = first_in_doc {
                    doc_terms.push(term_id);
                    doc_counts.push(1);
                }
            }

            // ---- per-document statistics ----
            if ntok == 0 {
                ttrs.push(0.0);
                max_tfs.push(0.0);
                mean_tfs.push(0.0);
                heaps_points.push((cum_tokens, vocab_seen));
                continue;
            }
            let uniq = doc_terms.len();
            let uniq_f = uniq as f64;
            ttrs.push(uniq_f / ntok as f64);
            mean_tfs.push(ntok as f64 / uniq_f);

            let mut max_tf = 0.0f64;
            let mut stop_here: u64 = 0;
            for (k, &term_id) in doc_terms.iter().enumerate() {
                let count = doc_counts[k] as u64;
                if is_stopword[term_id as usize] {
                    stop_here += count;
                } else if count as f64 > max_tf {
                    max_tf = count as f64;
                }
            }
            stopword_tokens += stop_here;
            max_tfs.push(max_tf);

            vocab_seen += new_terms;
            heaps_points.push((cum_tokens, vocab_seen));

            for (k, &term_id) in doc_terms.iter().enumerate() {
                let tf = doc_counts[k];
                postings[term_id as usize].push((i as u32, tf));
                vocab_freq[term_id as usize] += tf as u64;
            }
        }

        // Move the term strings out of the counting map into the lookup index.
        let mut term_index: FxHashMap<String, u32> = FxHashMap::default();
        term_index.reserve(counts.len());
        for (term, slot) in counts.into_iter() {
            term_index.insert(term, slot.term_id);
        }

        let avgdl = if n > 0 { total_tokens as f64 / n as f64 } else { 0.0 };
        Corpus {
            doc_ids,
            doc_len,
            term_index,
            postings,
            vocab_freq,
            ttrs,
            max_tfs,
            mean_tfs,
            heaps_points,
            stopword_tokens,
            total_tokens,
            avgdl,
        }
    }

    pub fn n_docs(&self) -> usize {
        self.doc_len.len()
    }

    pub fn vocab_size(&self) -> usize {
        self.postings.len()
    }

    pub fn term_id(&self, term: &str) -> Option<u32> {
        self.term_index.get(term).copied()
    }

    /// Document frequency == posting list length.
    pub fn df(&self, term: &str) -> u32 {
        match self.term_index.get(term) {
            Some(&id) => self.postings[id as usize].len() as u32,
            None => 0,
        }
    }

    pub fn doc_lens_f64(&self) -> Vec<f64> {
        self.doc_len.iter().map(|&l| l as f64).collect()
    }

    /// doc_id -> doc_idx, for joining qrels onto the index.
    pub fn doc_index_map(&self) -> FxHashMap<&str, u32> {
        let mut map: FxHashMap<&str, u32> = FxHashMap::default();
        map.reserve(self.doc_ids.len());
        for (i, id) in self.doc_ids.iter().enumerate() {
            map.insert(id.as_str(), i as u32);
        }
        map
    }
}
