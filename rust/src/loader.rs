//! Dataset loading, mirroring `autobm25/loader.py`:
//! standard format (docs/queries/qrels.jsonl), BEIR format
//! (corpus.jsonl + queries.jsonl + qrels/*.tsv) and streaming subsampling.

use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

use crate::json;

#[derive(Debug, Clone)]
pub struct Doc {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct Query {
    pub qid: String,
    pub query: String,
}

#[derive(Debug, Clone)]
pub struct Qrel {
    pub qid: String,
    pub doc_id: String,
    pub relevance: i32,
}

#[derive(Debug, Default)]
pub struct Dataset {
    pub docs: Vec<Doc>,
    pub queries: Vec<Query>,
    pub qrels: Vec<Qrel>,
}

pub fn exists(dir: &Path, name: &str) -> bool {
    dir.join(name).exists()
}

/// Read a file line by line, handing each non-empty line to `f`.
fn for_each_line<F>(path: &Path, mut f: F) -> Result<()>
where
    F: FnMut(&str) -> Result<()>,
{
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 12);
    loop {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf)?;
        if n == 0 {
            break;
        }
        while let Some(&last) = buf.last() {
            if last == b'\n' || last == b'\r' {
                buf.pop();
            } else {
                break;
            }
        }
        if buf.is_empty() {
            continue;
        }
        // Lines that are pure ASCII whitespace are skipped like Python's `line.strip()`.
        if buf.iter().all(|c| matches!(c, b' ' | b'\t' | b'\r')) {
            continue;
        }
        let line = String::from_utf8_lossy(&buf);
        f(&line)?;
    }
    Ok(())
}

fn parse_err(msg: String) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

fn field_text(fields: &[(String, json::Json)], key: &str) -> Option<String> {
    fields.iter().find(|(k, _)| k == key).and_then(|(_, v)| v.to_id())
}

/// Integer field; JSON numbers and numeric strings are both accepted, like
/// Python's `int(row["relevance"])`.
fn field_int(fields: &[(String, json::Json)], key: &str) -> i32 {
    match fields.iter().find(|(k, _)| k == key).map(|(_, v)| v) {
        Some(v) => {
            if let Some(i) = v.as_i64() {
                i as i32
            } else if let Some(s) = v.as_str() {
                s.trim().parse::<i32>().unwrap_or(0)
            } else {
                0
            }
        }
        None => 0,
    }
}

/// `{"_id": ..., "title": ..., "text": ...}` -> (id, text) with the title prefix
/// applied exactly like the Python loader does.
fn read_corpus_line(line: &str) -> Result<(String, String)> {
    let mut fields: Vec<(String, json::Json)> = Vec::with_capacity(3);
    json::scan_object(line, &["id", "_id", "title", "text"], &mut fields)
        .map_err(parse_err)?;
    let id = field_text(&fields, "_id")
        .or_else(|| field_text(&fields, "id"))
        .unwrap_or_default();
    let title = field_text(&fields, "title").unwrap_or_default();
    let text = field_text(&fields, "text").unwrap_or_default();
    let full = if !title.is_empty() { format!("{} {}", title, text) } else { text };
    Ok((id, full))
}

fn read_query_line(line: &str) -> Result<(String, String)> {
    let mut fields: Vec<(String, json::Json)> = Vec::with_capacity(4);
    json::scan_object(line, &["qid", "_id", "query", "text"], &mut fields)
        .map_err(parse_err)?;
    let qid = field_text(&fields, "qid")
        .or_else(|| field_text(&fields, "_id"))
        .unwrap_or_default();
    let query = field_text(&fields, "query")
        .or_else(|| field_text(&fields, "text"))
        .unwrap_or_default();
    Ok((qid, query))
}

/// Column variants accepted by the Python loader, checked in the same order.
fn col_index(header: &[String], variants: &[&str]) -> Option<usize> {
    for v in variants {
        for (i, h) in header.iter().enumerate() {
            if h == v {
                return Some(i);
            }
        }
    }
    None
}

fn tsv_files(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("tsv") {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// Read one qrels TSV file; pushes every row (positive and zero relevance alike).
fn read_qrels_tsv(path: &Path, out: &mut Vec<Qrel>, positive_only: bool) -> Result<()> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut header_line = String::new();
    if reader.read_line(&mut header_line)? == 0 {
        return Ok(());
    }
    let header: Vec<String> = header_line
        .trim()
        .to_lowercase()
        .split('\t')
        .map(|s| s.to_string())
        .collect();
    let i_q = col_index(&header, &["qid", "query-id", "query_id", "queryid"]);
    let i_d = col_index(&header, &["corpus_id", "corpus-id", "doc_id", "doc-id", "docid"]);
    let i_s = col_index(&header, &["score", "relevance", "rel"]);
    let (i_q, i_d, i_s) = match (i_q, i_d, i_s) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return Ok(()),
    };
    let need = i_q.max(i_d).max(i_s);
    let mut buf = String::new();
    loop {
        buf.clear();
        if reader.read_line(&mut buf)? == 0 {
            break;
        }
        let parts: Vec<&str> = buf.trim().split('\t').collect();
        if parts.len() <= need {
            continue;
        }
        let rel = match parts[i_s].trim().parse::<i32>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if positive_only && rel <= 0 {
            continue;
        }
        out.push(Qrel {
            qid: parts[i_q].to_string(),
            doc_id: parts[i_d].to_string(),
            relevance: rel,
        });
    }
    Ok(())
}

/// Load a dataset directory (standard or BEIR layout).
pub fn load_dataset(dir: &Path) -> Result<Dataset> {
    let mut ds = Dataset::default();

    let docs_path = dir.join("docs.jsonl");
    if docs_path.exists() {
        for_each_line(&docs_path, |line| {
            let (id, text) = read_corpus_line(line)?;
            ds.docs.push(Doc { id, text });
            Ok(())
        })?;
    } else {
        let corpus_path = dir.join("corpus.jsonl");
        if corpus_path.exists() {
            for_each_line(&corpus_path, |line| {
                let (id, text) = read_corpus_line(line)?;
                ds.docs.push(Doc { id, text });
                Ok(())
            })?;
        }
    }

    let queries_path = dir.join("queries.jsonl");
    if queries_path.exists() {
        for_each_line(&queries_path, |line| {
            let (qid, query) = read_query_line(line)?;
            ds.queries.push(Query { qid, query });
            Ok(())
        })?;
    }

    let qrels_path = dir.join("qrels.jsonl");
    if qrels_path.exists() {
        for_each_line(&qrels_path, |line| {
            let mut fields: Vec<(String, json::Json)> = Vec::with_capacity(3);
            json::scan_object(line, &["qid", "doc_id", "relevance"], &mut fields)
                .map_err(parse_err)?;
            ds.qrels.push(Qrel {
                qid: field_text(&fields, "qid").unwrap_or_default(),
                doc_id: field_text(&fields, "doc_id").unwrap_or_default(),
                relevance: field_int(&fields, "relevance"),
            });
            Ok(())
        })?;
    } else {
        for path in tsv_files(&dir.join("qrels"))? {
            read_qrels_tsv(&path, &mut ds.qrels, true)?;
        }
    }
    Ok(ds)
}

/// Tiny xorshift64* generator; used only for reservoir sampling of non-relevant docs.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next_u64() % n
        }
    }
}

/// Streaming load + stratified subsampling for huge corpora: every judged-relevant
/// document is kept, the rest is reservoir sampled down to `max_docs`.
///
/// Note: the Python version draws the reservoir with `random.Random(42)`; the Rust
/// version uses its own generator, so *which* non-relevant documents are kept can
/// differ. Relevant documents, queries and qrels are identical.
pub fn load_dataset_subsampled(dir: &Path, max_docs: usize, seed: u64) -> Result<Dataset> {
    let mut ds = Dataset::default();

    // 1. all qrels + the set of judged-relevant doc ids
    let mut qrels_all: Vec<Qrel> = Vec::new();
    let qrels_path = dir.join("qrels.jsonl");
    if qrels_path.exists() {
        for_each_line(&qrels_path, |line| {
            let mut fields: Vec<(String, json::Json)> = Vec::with_capacity(3);
            json::scan_object(line, &["qid", "doc_id", "relevance"], &mut fields)
                .map_err(parse_err)?;
            qrels_all.push(Qrel {
                qid: field_text(&fields, "qid").unwrap_or_default(),
                doc_id: field_text(&fields, "doc_id").unwrap_or_default(),
                relevance: field_int(&fields, "relevance"),
            });
            Ok(())
        })?;
    } else {
        for path in tsv_files(&dir.join("qrels"))? {
            read_qrels_tsv(&path, &mut qrels_all, false)?;
        }
    }
    let mut relevant_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for r in &qrels_all {
        if r.relevance > 0 {
            relevant_ids.insert(r.doc_id.clone());
        }
    }

    // 2. stream the corpus, keep relevant docs, reservoir sample the rest
    let mut corpus_path = dir.join("docs.jsonl");
    if !corpus_path.exists() {
        corpus_path = dir.join("corpus.jsonl");
    }
    let budget = max_docs.saturating_sub(relevant_ids.len().min(max_docs));
    let mut rel_count = 0usize;
    let mut nonrel_seen: u64 = 0;
    let mut nonrel: Vec<Doc> = Vec::with_capacity(budget.min(1 << 16));
    let mut rng = Rng(seed ^ 0x9E37_79B9_7F4A_7C15);
    for_each_line(&corpus_path, |line| {
        let (id, text) = read_corpus_line(line)?;
        if relevant_ids.contains(&id) {
            if rel_count < max_docs {
                rel_count += 1;
                ds.docs.push(Doc { id, text });
            }
        } else if budget > 0 {
            nonrel_seen += 1;
            if nonrel.len() < budget {
                nonrel.push(Doc { id, text });
            } else {
                let j = rng.below(nonrel_seen);
                if j < budget as u64 {
                    nonrel[j as usize] = Doc { id, text };
                }
            }
        }
        Ok(())
    })?;
    ds.docs.extend(nonrel);
    ds.docs.truncate(max_docs);

    // 3. queries and the filtered qrels
    let queries_path = dir.join("queries.jsonl");
    if queries_path.exists() {
        for_each_line(&queries_path, |line| {
            let (qid, query) = read_query_line(line)?;
            ds.queries.push(Query { qid, query });
            Ok(())
        })?;
    }
    let valid: std::collections::HashSet<&str> =
        ds.docs.iter().map(|d| d.id.as_str()).collect();
    for r in qrels_all {
        if r.relevance > 0 && valid.contains(r.doc_id.as_str()) {
            ds.qrels.push(r);
        }
    }
    Ok(ds)
}
