//! Tokenizer + stopwords, byte-compatible with the Python implementation
//! (`text.lower().split()`).
//!
//! Python lowercases the whole string per character and then splits on runs of
//! Unicode whitespace. We do the same, but into a reusable buffer so that the
//! hot loop allocates nothing.

use std::collections::HashSet;
use std::path::Path;

/// `str.isspace()` in Python also treats the file/group/record/unit separators as
/// whitespace, which `char::is_whitespace` does not.
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{1f}')
}

/// Reusable lowercase + split buffer.
///
/// `spans` holds byte ranges into `buf`; `tokens()` maps them back to `&str`.
pub struct TokenBuffer {
    pub buf: String,
    pub spans: Vec<(usize, usize)>,
}

impl TokenBuffer {
    pub fn new() -> TokenBuffer {
        TokenBuffer { buf: String::new(), spans: Vec::new() }
    }

    /// Lowercase `text` and split it into whitespace-separated tokens.
    pub fn refill(&mut self, text: &str) {
        self.buf.clear();
        self.spans.clear();
        if text.is_ascii() {
            // Fast path for the overwhelmingly common ASCII case: one byte test per
            // character instead of the Unicode lowercase tables.
            self.buf.reserve(text.len());
            for byte in text.bytes() {
                self.buf.push(byte.to_ascii_lowercase() as char);
            }
        } else {
            for ch in text.chars() {
                // `char::to_lowercase` is the per-character full lowercase mapping,
                // which is exactly what Python's `str.lower()` applies.
                for lower in ch.to_lowercase() {
                    self.buf.push(lower);
                }
            }
        }
        let mut start: Option<usize> = None;
        for (idx, ch) in self.buf.char_indices() {
            if is_py_space(ch) {
                if let Some(s) = start.take() {
                    self.spans.push((s, idx));
                }
            } else if start.is_none() {
                start = Some(idx);
            }
        }
        if let Some(s) = start {
            self.spans.push((s, self.buf.len()));
        }
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn get(&self, i: usize) -> &str {
        let (s, e) = self.spans[i];
        &self.buf[s..e]
    }

    pub fn tokens(&self) -> TokenIter<'_> {
        TokenIter { buf: self.buf.as_str(), spans: self.spans.as_slice(), i: 0 }
    }

    /// Tokens in first-occurrence order, with their in-query frequency.
    pub fn counted(&self) -> (Vec<&str>, Vec<u32>) {
        let mut terms: Vec<&str> = Vec::with_capacity(self.spans.len());
        let mut counts: Vec<u32> = Vec::with_capacity(self.spans.len());
        let mut index: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::with_capacity(self.spans.len() * 2);
        for tok in self.tokens() {
            match index.get(tok) {
                Some(&slot) => counts[slot] += 1,
                None => {
                    index.insert(tok, terms.len());
                    terms.push(tok);
                    counts.push(1);
                }
            }
        }
        (terms, counts)
    }
}

impl Default for TokenBuffer {
    fn default() -> Self {
        TokenBuffer::new()
    }
}

pub struct TokenIter<'a> {
    buf: &'a str,
    spans: &'a [(usize, usize)],
    i: usize,
}

impl<'a> Iterator for TokenIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.i >= self.spans.len() {
            return None;
        }
        let (s, e) = self.spans[self.i];
        self.i += 1;
        Some(&self.buf[s..e])
    }
}

/// One token as an owned `String` (convenience for callers that outlive a buffer).
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tb = TokenBuffer::new();
    tb.refill(text);
    tb.tokens().map(|t| t.to_string()).collect()
}

/// Read the bundled stopword list (one word per line, blank lines skipped).
pub fn load_stopwords(path: &Path) -> std::io::Result<HashSet<String>> {
    let raw = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&raw);
    let mut set = HashSet::new();
    for line in text.lines() {
        let w = line.trim();
        if !w.is_empty() {
            set.insert(w.to_string());
        }
    }
    Ok(set)
}
