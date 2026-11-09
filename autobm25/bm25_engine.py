"""Self-contained BM25 / BM25+ retrieval engine (inverted index).

Scoring formula:
  BM25:  score += idf(q) * [f(q,D)*(k1+1) / (f(q,D) + k1*(1-b+b*|D|/avgdl))]
                 * [f(q,Q)*(k3+1) / (f(q,Q) + k3)]   (query-TF saturation, 1 when k3=0)
  BM25+: score += idf(q) * [the term above + delta]

Two IDF variants:
  rsj:      ln((N - df + 0.5) / (df + 0.5))
  smoothed: ln((N + 1) / (df + 1))

The class below is the pure-Python implementation *and* the front end of an
optional Rust backend (see `autobm25/_accel.py`): when the PyO3 extension is
importable, `build_index` / `search` / `search_many` / `features` hand the work to
Rust and only the small bookkeeping attributes stay in Python. The public API and
the results are identical either way; `explain()` reports which one is live.
"""

import math
import operator

import numpy as np

from . import _accel


def _top_k_value(top_k):
    """Validate a public result limit without silently accepting floats."""
    try:
        value = operator.index(top_k)
    except TypeError as exc:
        raise TypeError("top_k must be an integer") from exc
    if value < 0:
        raise ValueError("top_k must be greater than or equal to zero")
    return value


class _TermCountView:
    """Stands in for `BM25Engine.postings` on the Rust backend.

    The Rust index keeps its posting lists in a flat table, so instead of mirroring
    them back into Python dicts this exposes just what the attribute is normally
    used for (`len(engine.postings)`, truthiness).
    """

    __slots__ = ("_n",)

    def __init__(self, n):
        self._n = int(n)

    def __len__(self):
        return self._n

    def __bool__(self):
        return self._n > 0

    def __repr__(self):
        return f"<rust postings: {self._n} terms>"


def tokenize(text):
    """Whitespace tokenization + lowercasing (stopwords are not removed).

    Delegates to Rust when available; the Rust tokenizer is byte-compatible with
    `text.lower().split()` (including Python's extra separator characters).
    """
    if _accel.ENABLED:
        return _accel.rs.tokenize(text)
    return text.lower().split()


class BM25Engine:
    def __init__(self, k1=1.2, b=0.75, k3=0.0, delta=0.0, idf_type="rsj"):
        self.k1 = k1
        self.b = b
        self.k3 = k3
        self.delta = delta
        self.idf_type = idf_type
        self.N = 0
        self.avgdl = 0.0
        self.doc_ids = []     # doc_idx -> doc_id
        self.doc_len = None   # np.ndarray[int32] on the Python backend, None on Rust
        self.postings = {}    # term -> (doc_idx array, tf array)
        self.doc_freq = {}    # term -> df
        self.idf_cache = {}
        self._rust = None     # autobm25._autobm25_rs.Index when the Rust backend is live

    # ----------------------------------------------------------------- backend

    @property
    def uses_rust(self):
        """Whether this engine is backed by the Rust extension."""
        return self._rust is not None

    def backend(self):
        """`"rust"` or `"python"`, for logging and benchmark reports."""
        return "rust" if self._rust is not None else "python"

    def explain(self):
        """Human-readable description of the live backend."""
        if self._rust is not None:
            return f"rust (PyO3 extension {_accel.rs.__version__})"
        if not _accel.ENABLED:
            info = _accel.backend_info()
            return f"python (Rust extension unavailable: {info.get('reason', 'unknown')})"
        return "python (Rust disabled via AUTOBM25_NO_RUST)"

    # ------------------------------------------------------------------- index

    def build_index(self, docs):
        """docs: [{"id": str, "text": str}]"""
        if _accel.ENABLED:
            return self._build_index_rust(docs)
        return self._build_index_python(docs)

    def _build_index_rust(self, docs):
        """One Rust pass: tokenize, index, and collect the feature statistics.

        Where the Python path walks the corpus here and then walks it again inside
        `extract_features`, the Rust path tokenizes each document exactly once and
        keeps everything both stages need.
        """
        doc_ids = [str(d["id"]) for d in docs]
        self._rust = _accel.Index(
            doc_ids, [d["text"] for d in docs], _accel.stopword_list()
        )
        self.doc_ids = doc_ids
        self.N = self._rust.n_docs
        self.avgdl = self._rust.avgdl
        # `doc_len` / `_dl_float` stay None: the Rust index owns the arrays and no
        # score round-trips through Python. Use `doc_lengths()` to read them back.
        self.doc_len = None
        self._dl_float = None
        self.postings = _TermCountView(self._rust.n_terms)
        self.doc_freq = {}
        self.idf_cache = {}
        return self

    def _build_index_python(self, docs):
        self.doc_ids = [d["id"] for d in docs]
        self.N = len(docs)
        self.doc_len = np.zeros(self.N, dtype=np.int32)
        total_len = 0
        tmp_postings = {}  # term -> [(doc_idx, tf), ...]
        for i, d in enumerate(docs):
            toks = tokenize(d["text"])
            self.doc_len[i] = len(toks)
            total_len += self.doc_len[i]
            seen = {}
            for t in toks:
                seen[t] = seen.get(t, 0) + 1
            for t, tf in seen.items():
                tmp_postings.setdefault(t, []).append((i, tf))
        self.avgdl = total_len / self.N if self.N else 0.0
        self._dl_float = self.doc_len.astype(np.float64)
        for t, pairs in tmp_postings.items():
            arr = np.asarray(pairs, dtype=np.int64)
            self.postings[t] = (arr[:, 0], arr[:, 1])
        self.doc_freq = {t: len(docs_arr) for t, (docs_arr, _) in self.postings.items()}
        self.idf_cache = {}
        return self

    # ------------------------------------------------------------------ params

    def set_params(self, k1=None, b=None, k3=None, delta=None, idf_type=None):
        if k1 is not None:
            self.k1 = k1
        if b is not None:
            self.b = b
        if k3 is not None:
            self.k3 = k3
        if delta is not None:
            self.delta = delta
        if idf_type is not None and idf_type != self.idf_type:
            self.idf_type = idf_type
            self.idf_cache = {}
        if self._rust is not None:
            self._rust.set_params(k1=k1, b=b, k3=k3, delta=delta, idf_type=idf_type)

    def idf(self, term):
        if self._rust is not None:
            return self._rust.idf(term)
        if term not in self.idf_cache:
            df = self.doc_freq.get(term, 0)
            N = self.N
            if self.idf_type == "rsj":
                val = math.log((N - df + 0.5) / (df + 0.5))
            else:  # smoothed
                val = math.log((N + 1) / (df + 1))
            self.idf_cache[term] = val
        return self.idf_cache[term]

    def score_doc(self, term, doc_idx, idf_val):
        """Score a single document (kept for debugging; batch retrieval uses
        `search`)."""
        if self._rust is not None:
            return self._rust.score_doc(term, int(doc_idx), float(idf_val))
        docs_arr, tf_arr = self.postings[term]
        pos = int(np.searchsorted(docs_arr, doc_idx)) if docs_arr[0] <= doc_idx else -1
        if pos >= len(docs_arr) or docs_arr[pos] != doc_idx:
            return 0.0
        tf = int(tf_arr[pos])
        dl = int(self.doc_len[doc_idx])
        k1, b, avgdl = self.k1, self.b, self.avgdl
        denom = tf + k1 * (1 - b + b * dl / avgdl) if avgdl > 0 else tf + k1
        tf_part = tf * (k1 + 1) / denom
        if self.delta > 0:
            return idf_val * (tf_part + self.delta)
        return idf_val * tf_part

    # ------------------------------------------------------------------ search

    def search(self, query, top_k=10):
        """Return [(doc_id, score), ...] sorted by descending score."""
        top_k = _top_k_value(top_k)
        if self._rust is not None:
            return self._rust.search(query, top_k)
        if self.N == 0 or top_k == 0:
            return []
        k1, b, k3, delta, avgdl = self.k1, self.b, self.k3, self.delta, self.avgdl
        dl = self._dl_float
        scores = np.zeros(self.N, dtype=np.float64)
        qtf = {}
        for t in tokenize(query):
            qtf[t] = qtf.get(t, 0) + 1
        for t, qcount in qtf.items():
            post = self.postings.get(t)
            if post is None:
                continue
            doc_idx, tf = post
            idf_val = self.idf(t)
            # Vectorized BM25 TF-saturation term: idf * f*(k1+1) / (f + k1*(1-b+b*dl/avgdl))
            denom = tf + k1 * (1 - b + b * dl[doc_idx] / avgdl) if avgdl > 0 else tf + k1
            part = idf_val * tf * (k1 + 1) / denom
            if k3 > 0 and qcount > 1:  # query-TF saturation: f(q,Q)*(k3+1)/(f(q,Q)+k3)
                part = part * (qcount * (k3 + 1) / (qcount + k3))
            if delta > 0:  # BM25+ compensation term
                part = part + idf_val * delta
            np.add.at(scores, doc_idx, part)
        nonzero = np.flatnonzero(scores)
        if nonzero.size == 0:
            return []
        k = min(top_k, nonzero.size)
        top = nonzero[np.argpartition(scores[nonzero], -k)[-k:]]
        # Sort by score descending, doc_idx ascending (deterministic on ties)
        order = top[np.lexsort((top, -scores[top]))]
        return [(self.doc_ids[int(i)], float(scores[i])) for i in order]

    def search_many(self, queries, top_k=10, threads=None):
        """Search a batch of queries, returning one ranked list per query.

        On the Rust backend this runs with the GIL released and one scorer per
        worker thread, so a multi-thousand-query evaluation no longer serializes on
        Python's per-query overhead. On the Python backend it is a plain loop.
        """
        top_k = _top_k_value(top_k)
        if self._rust is not None:
            return self._rust.search_many(queries, top_k, threads)
        return [self.search(q, top_k) for q in queries]

    # ---------------------------------------------------------------- features

    def features(self, queries=()):
        """The corpus feature dict, read off the pass that built this index.

        Rust backend only - the Python backend keeps no feature statistics, so call
        `feature_extractor.extract_features(docs, queries)` there.
        """
        if self._rust is None:
            raise RuntimeError(
                "features() needs the Rust backend; use "
                "autobm25.feature_extractor.extract_features(docs, queries) instead"
            )
        return self._rust.features(
            [q["query"] if isinstance(q, dict) else q for q in queries]
        )

    def doc_lengths(self):
        """Token counts per document, in index order."""
        if self._rust is not None:
            return self._rust.doc_lengths()
        return [int(n) for n in self.doc_len]
