//! AutoBM25 core, shared by the standalone `autobm25-rs` CLI and by the PyO3
//! extension module (`--features python`).
//!
//! The module layout mirrors the Python package one-for-one:
//!
//! | Python (`autobm25/`)  | Rust (`rust/src/`) |
//! |-----------------------|--------------------|
//! | `bm25_engine.py`      | `corpus.rs`, `engine.rs` |
//! | `feature_extractor.py`| `features.rs`, `stats.rs` |
//! | `dictionary.py`       | `params.rs` (dictionary half) |
//! | `rule_predictor.py`   | `params.rs` (heuristic half) |
//! | `loader.py`           | `loader.rs`, `json.rs` |
//! | `metrics.py`          | `metrics.rs` |

pub mod corpus;
pub mod engine;
pub mod features;
pub mod fx;
pub mod grid;
pub mod json;
pub mod loader;
pub mod metrics;
pub mod params;
pub mod stats;
pub mod text;

#[cfg(feature = "python")]
mod pyext;
