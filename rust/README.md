# AutoBM25 — Rust core

A dependency-free Rust port of the `autobm25` package's retrieval core, plus the
PyO3 layer that exposes it to Python.

| file | what it is |
|---|---|
| `src/lib.rs` and friends | the core library: corpus build, features, dictionary lookup, metrics, grid search |
| `src/pyext.rs` | PyO3 bindings, compiled into the `autobm25._autobm25_rs` extension module |
| `src/main.rs` | the standalone `autobm25-rs` CLI (batch / interactive / features / eval / grid), no Python involved |

## Building

```bash
cargo build --release          # standalone CLI + library; no third-party crates
python build_pyext.py          # the Python extension (pulls in pyo3)
python build_pyext.py --check  # ... and verify it imports
```

`build_pyext.py` copies the cdylib into the package as
`autobm25/_autobm25_rs.pyd`, where `autobm25/_accel.py` picks it up
automatically. Without it the Python package runs its pure-Python
implementation, so building this is optional.

You normally do not call it by hand: the repository's `setup.py` runs it during
`pip install .` / `pip wheel .` when the extension is missing and `cargo` is on
PATH, then tags the resulting wheel `cp39-abi3-<platform>`. **End users of that
wheel never need Rust** — the compiled extension is inside the wheel, and
`pip install` of a source checkout without a toolchain still yields a working
pure-Python package.

Toolchains are tried in order — the default first, then any installed
`*-windows-gnu` one — because a default `stable-x86_64-pc-windows-msvc` without
the Visual C++ linker fails at link time while the MinGW toolchain beside it
works. Pass `--toolchain stable-x86_64-pc-windows-gnu` to pin one explicitly.

## Module map

| Python (`autobm25/`) | Rust (`src/`) |
|---|---|
| `bm25_engine.py` | `corpus.rs`, `engine.rs` |
| `feature_extractor.py` | `features.rs`, `stats.rs` |
| `dictionary.py` | `params.rs` (dictionary half) |
| `rule_predictor.py` | `params.rs` (heuristic half) |
| `loader.py` | `loader.rs`, `json.rs` |
| `metrics.py` | `metrics.rs` |
| `_accel.py` | `pyext.rs` |
| — | `grid.rs`, `fx.rs`, `text.rs` |

## Verifying parity

From the repository root:

```bash
python _bench/bench_backend.py dataset_en/scifact --compare  # Python vs the PyO3 build
python _bench/compare.py dataset_en/scifact                  # Python vs the standalone binary
python _bench/_diag_ties.py dataset_en/nfcorpus 100          # characterise ranking tie-breaks
```

The Python side runs at ~0.75M tokens/s while this core builds an index at ~7.2M
tokens/s; see the "Rust acceleration (PyO3)" section of the top-level README for
the full measurements and the parity caveats.
