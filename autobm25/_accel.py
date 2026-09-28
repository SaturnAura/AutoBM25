"""Optional Rust acceleration for the AutoBM25 core.

The three stages that dominate a Python run are implemented in Rust with PyO3
(see `rust/`):

* tokenizing the corpus and building the inverted index,
* the per-document statistics behind the feature vector,
* BM25 / BM25+ scoring over the posting lists.

When the compiled extension (`autobm25._autobm25_rs`) is importable it is used
transparently; when it is not, every entry point falls back to the pure-Python
implementation, so `pip install .` on its own keeps working and nothing here is
required for correctness.

Set ``AUTOBM25_NO_RUST=1`` to force the Python path - that is how
`_bench/compare.py` and `_bench/bench_backend.py` compare the two backends.
"""

import importlib
import os
import threading

_DISABLE_ENV = "AUTOBM25_NO_RUST"
_TRUTHY = ("1", "true", "yes", "on")
_EXTENSION = f"{__package__}._autobm25_rs"

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STOPWORDS_FILE = os.path.join(_DATA_DIR, "stopwords.txt")


def disabled_by_env():
    """Whether `AUTOBM25_NO_RUST` asks for the pure-Python backend."""
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in _TRUTHY


_load_error = None
rs = None
if not disabled_by_env():
    try:
        # `importlib` rather than `from . import ...`: this module is imported from
        # `autobm25/__init__.py`, so the package is still initializing and the
        # attribute-based form would report a bogus circular import.
        rs = importlib.import_module(_EXTENSION)
    except ImportError as exc:  # pragma: no cover - depends on the local build
        rs = None
        text = str(exc)
        if "No module named" in text or "cannot import name" in text:
            _load_error = (
                f"the compiled extension {_EXTENSION} is not built "
                f"(run `python rust/build_pyext.py`)"
            )
        else:
            _load_error = f"{type(exc).__name__}: {text}"

ENABLED = rs is not None
BACKEND = "rust" if ENABLED else "python"

#: The Rust `Index` class, or `None` when the extension is not available.
Index = rs.Index if ENABLED else None

_stopwords = None
_stopwords_lock = threading.Lock()


def stopword_list():
    """The bundled stopword list, cached, in file order."""
    global _stopwords
    if _stopwords is None:
        with _stopwords_lock:
            if _stopwords is None:
                with open(STOPWORDS_FILE, encoding="utf-8") as f:
                    _stopwords = [w.strip() for w in f if w.strip()]
    return _stopwords


def stopword_set():
    """The bundled stopwords as a `set` (the shape `extract_features` wants)."""
    return set(stopword_list())


def extract_features(docs, queries=None, stopwords=None):
    """Rust `extract_features`: same arguments, same dict as the Python version.

    `docs` is `[{"id", "text"}]` and `queries` is `[{"qid", "query"}]`, exactly as
    the Python function takes them.
    """
    if not ENABLED:  # pragma: no cover - guarded by the callers
        raise RuntimeError("the Rust extension is not available")
    stop = stopword_list() if stopwords is None else list(stopwords)
    return rs.extract_features(
        [d["text"] for d in docs],
        [q["query"] for q in (queries or [])],
        stop,
        [str(d["id"]) for d in docs],
    )


def backend_info():
    """Diagnostics for `--version`-style reporting and the README numbers."""
    if not ENABLED:
        reason = (
            f"disabled by {_DISABLE_ENV}"
            if disabled_by_env()
            else (_load_error or "extension not built")
        )
        return {"backend": "python", "available": False, "reason": reason}
    return {
        "backend": "rust",
        "available": True,
        "version": rs.__version__,
        "module": rs.__name__,
        "path": getattr(rs, "__file__", None),
    }
