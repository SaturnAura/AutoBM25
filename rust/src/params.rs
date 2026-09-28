//! Hyperparameters: the parameter dictionary (O(1) lookup + OOD guard), the
//! heuristic rule fallback, and the YAML-subset config reader.

use std::collections::HashMap;
use std::cmp::Ordering;
use std::path::Path;

use crate::features::Features;
use crate::json::{self, Json};
use crate::stats;

pub const DEFAULT_FEATURE_KEYS: [&str; 21] = [
    "avgdl",
    "std_len",
    "cv_len",
    "len_skew",
    "len_p90",
    "len_p10",
    "len_ratio_p90_p10",
    "avg_ttr",
    "avg_max_tf",
    "length_tf_corr",
    "hapax_ratio",
    "vocab_size",
    "heaps_beta",
    "stopword_density",
    "zipf_alpha",
    "avg_query_len",
    "avg_query_max_tf",
    "query_repeat_ratio",
    "query_idf_mean",
    "query_idf_std",
    "query_oov_ratio",
];

#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    pub k1: f64,
    pub b: f64,
    pub k3: f64,
    pub delta: f64,
    pub idf_type: String,
}

impl Params {
    pub fn model_variant(&self) -> &'static str {
        if self.delta > 0.0 {
            "bm25+"
        } else {
            "bm25"
        }
    }

    pub fn to_json(&self) -> Json {
        json::obj(vec![
            ("k1", Json::num(self.k1)),
            ("b", Json::num(self.b)),
            ("k3", Json::num(self.k3)),
            ("delta", Json::num(self.delta)),
            ("idf_type", Json::str(self.idf_type.clone())),
            ("model_variant", Json::str(self.model_variant())),
        ])
    }

    /// `"k1=1.2 b=0.75 ..."` for log lines.
    pub fn describe(&self) -> String {
        format!(
            "k1={} b={} k3={} delta={} idf={} ({})",
            self.k1, self.b, self.k3, self.delta, self.idf_type, self.model_variant()
        )
    }
}

// ------------------------------------------------------------------ config

#[derive(Debug, Clone)]
pub struct Coefficients {
    pub k1_base: f64,
    pub alpha_ttr: f64,
    pub alpha_maxtf: f64,
    pub b_base: f64,
    pub beta_corr: f64,
    pub beta_cv: f64,
    pub gamma_delta: f64,
    pub k3_base: f64,
    pub k3_rep: f64,
    pub heaps_beta_threshold: f64,
    pub hapax_ratio_threshold: f64,
    pub k1_min: f64,
    pub k1_max: f64,
    pub b_min: f64,
    pub b_max: f64,
    pub delta_min: f64,
    pub delta_max: f64,
    pub k3_min: f64,
    pub k3_max: f64,
}

impl Default for Coefficients {
    fn default() -> Self {
        Coefficients {
            k1_base: 1.5,
            alpha_ttr: 1.5,
            alpha_maxtf: 0.3,
            b_base: 0.5,
            beta_corr: 0.4,
            beta_cv: 0.3,
            gamma_delta: 1.0,
            k3_base: 0.5,
            k3_rep: 2.0,
            heaps_beta_threshold: 0.7,
            hapax_ratio_threshold: 0.6,
            k1_min: 0.5,
            k1_max: 3.0,
            b_min: 0.0,
            b_max: 1.0,
            delta_min: 0.0,
            delta_max: 2.0,
            k3_min: 0.0,
            k3_max: 8.0,
        }
    }
}

impl Coefficients {
    fn set(&mut self, key: &str, value: f64) {
        match key {
            "k1_base" => self.k1_base = value,
            "alpha_ttr" => self.alpha_ttr = value,
            "alpha_maxtf" => self.alpha_maxtf = value,
            "b_base" => self.b_base = value,
            "beta_corr" => self.beta_corr = value,
            "beta_cv" => self.beta_cv = value,
            "gamma_delta" => self.gamma_delta = value,
            "k3_base" => self.k3_base = value,
            "k3_rep" => self.k3_rep = value,
            "heaps_beta_threshold" => self.heaps_beta_threshold = value,
            "hapax_ratio_threshold" => self.hapax_ratio_threshold = value,
            "k1_min" => self.k1_min = value,
            "k1_max" => self.k1_max = value,
            "b_min" => self.b_min = value,
            "b_max" => self.b_max = value,
            "delta_min" => self.delta_min = value,
            "delta_max" => self.delta_max = value,
            "k3_min" => self.k3_min = value,
            "k3_max" => self.k3_max = value,
            _ => {}
        }
    }
}

#[derive(Debug, Clone)]
pub struct GridSpace {
    pub k1: Vec<f64>,
    pub b: Vec<f64>,
    pub k3: Vec<f64>,
    pub delta: Vec<f64>,
    pub idf_type: Vec<String>,
    pub metric: String,
    pub top_k: usize,
}

impl Default for GridSpace {
    fn default() -> Self {
        GridSpace {
            k1: vec![0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
            b: vec![0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
            k3: vec![0.0, 0.5, 1.2, 2.0, 4.0, 8.0],
            delta: vec![0.0, 0.3, 0.5, 0.7, 1.0, 1.5],
            idf_type: vec!["rsj".to_string(), "smoothed".to_string()],
            metric: "ndcg@10".to_string(),
            top_k: 100,
        }
    }
}

impl GridSpace {
    fn set(&mut self, key: &str, value: &str) {
        match key {
            "k1" => self.k1 = parse_list_f64(value, &self.k1),
            "b" => self.b = parse_list_f64(value, &self.b),
            "k3" => self.k3 = parse_list_f64(value, &self.k3),
            "delta" => self.delta = parse_list_f64(value, &self.delta),
            "idf_type" => self.idf_type = parse_list_str(value, &self.idf_type),
            "metric" => self.metric = unquote(value).to_string(),
            "top_k" => {
                if let Some(v) = parse_number(value) {
                    self.top_k = v as usize;
                }
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub coefficients: Coefficients,
    pub default_params: Params,
    pub grid: GridSpace,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            coefficients: Coefficients::default(),
            default_params: Params {
                k1: 1.2,
                b: 0.75,
                k3: 0.0,
                delta: 0.0,
                idf_type: "rsj".to_string(),
            },
            grid: GridSpace::default(),
        }
    }
}

fn unquote(s: &str) -> &str {
    let t = s.trim();
    if t.len() >= 2 && ((t.starts_with('"') && t.ends_with('"')) || (t.starts_with('\'') && t.ends_with('\''))) {
        &t[1..t.len() - 1]
    } else {
        t
    }
}

fn parse_number(s: &str) -> Option<f64> {
    unquote(s).trim().parse::<f64>().ok()
}

fn parse_list_f64(value: &str, defaults: &[f64]) -> Vec<f64> {
    let inner = value.trim().trim_start_matches('[').trim_end_matches(']');
    let out: Vec<f64> = inner.split(',').filter_map(|p| parse_number(p)).collect();
    if out.is_empty() {
        defaults.to_vec()
    } else {
        out
    }
}

fn parse_list_str(value: &str, defaults: &[String]) -> Vec<String> {
    let inner = value.trim().trim_start_matches('[').trim_end_matches(']');
    let out: Vec<String> = inner
        .split(',')
        .map(|p| unquote(p).trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if out.is_empty() {
        defaults.to_vec()
    } else {
        out
    }
}

/// Drop a trailing `# comment` (quotes are respected).
fn strip_comment(line: &str) -> &str {
    let mut quote: Option<char> = None;
    for (i, ch) in line.char_indices() {
        match quote {
            Some(q) => {
                if ch == q {
                    quote = None;
                }
            }
            None => {
                if ch == '"' || ch == '\'' {
                    quote = Some(ch);
                } else if ch == '#' {
                    return &line[..i];
                }
            }
        }
    }
    line
}

/// Read the flat subset of `config.yaml` this project uses. Unknown keys, comments
/// and non-UTF-8 comment bytes are ignored, so an oddly encoded config still works.
pub fn load_config(path: &Path) -> Config {
    let mut cfg = Config::default();
    let raw = match std::fs::read(path) {
        Ok(r) => r,
        Err(_) => return cfg,
    };
    let text = String::from_utf8_lossy(&raw);
    let mut section = String::new();
    for line in text.lines() {
        let stripped = strip_comment(line);
        let trimmed = stripped.trim_end();
        if trimmed.trim().is_empty() {
            continue;
        }
        let indent = trimmed.len() - trimmed.trim_start().len();
        let body = trimmed.trim_start();
        if body.starts_with('-') {
            continue;
        }
        let (key, value) = match body.split_once(':') {
            Some(kv) => kv,
            None => continue,
        };
        let key = key.trim();
        let value = value.trim();
        if value.is_empty() {
            if indent == 0 {
                section = key.to_string();
            }
            continue;
        }
        if indent == 0 {
            continue;
        }
        match section.as_str() {
            "coefficients" => {
                if let Some(v) = parse_number(value) {
                    cfg.coefficients.set(key, v);
                }
            }
            "default_params" => match key {
                "k1" => {
                    if let Some(v) = parse_number(value) {
                        cfg.default_params.k1 = v;
                    }
                }
                "b" => {
                    if let Some(v) = parse_number(value) {
                        cfg.default_params.b = v;
                    }
                }
                "k3" => {
                    if let Some(v) = parse_number(value) {
                        cfg.default_params.k3 = v;
                    }
                }
                "delta" => {
                    if let Some(v) = parse_number(value) {
                        cfg.default_params.delta = v;
                    }
                }
                "idf_type" => cfg.default_params.idf_type = unquote(value).to_string(),
                _ => {}
            },
            "grid_search" => cfg.grid.set(key, value),
            _ => {}
        }
    }
    cfg
}

// -------------------------------------------------------------- heuristics

fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Heuristic rules from the paper/config: features -> hyperparameters.
pub fn predict(f: &Features, cfg: &Config) -> Params {
    let c = &cfg.coefficients;
    let k1 = clip(
        c.k1_base - c.alpha_ttr * f.get_or("avg_ttr", 0.0)
            + c.alpha_maxtf * (f.get_or("avg_max_tf", 0.0) / 20.0).min(1.0),
        c.k1_min,
        c.k1_max,
    );
    let b = clip(
        c.b_base + c.beta_corr * f.get_or("length_tf_corr", 0.0)
            - c.beta_cv * f.get_or("cv_len", 0.0),
        c.b_min,
        c.b_max,
    );
    let delta = clip(c.gamma_delta * f.get_or("cv_len", 0.0), c.delta_min, c.delta_max);
    let k3 = clip(
        c.k3_base * (f.get_or("avg_query_max_tf", 1.0) - 1.0).max(0.0)
            + c.k3_rep * f.get_or("query_repeat_ratio", 0.0),
        c.k3_min,
        c.k3_max,
    );
    let idf_type = if f.get_or("heaps_beta", 0.0) > c.heaps_beta_threshold
        || f.get_or("hapax_ratio", 0.0) > c.hapax_ratio_threshold
    {
        "smoothed"
    } else {
        "rsj"
    };
    Params {
        k1: stats::round_digits(k1, 4),
        b: stats::round_digits(b, 4),
        k3: stats::round_digits(k3, 4),
        delta: stats::round_digits(delta, 4),
        idf_type: idf_type.to_string(),
    }
}

// -------------------------------------------------------------- dictionary

pub struct Entry {
    pub name: String,
    pub features: Vec<(String, f64)>,
    pub params: Params,
    pub vector: Vec<(String, f64)>,
}

pub struct DictLookup {
    pub params: Params,
    pub distance: f64,
    pub matches: Vec<String>,
}

pub struct ParamDictionary {
    pub feature_keys: Vec<String>,
    pub entries: Vec<Entry>,
    pub mean: HashMap<String, f64>,
    pub std: HashMap<String, f64>,
    pub fmin: HashMap<String, f64>,
    pub fmax: HashMap<String, f64>,
    pub k_neighbors: usize,
    pub match_threshold: f64,
    pub normalize: bool,
    pub reject_oov: bool,
    pub oov_margin: f64,
    pub oov_min_features: usize,
}

impl ParamDictionary {
    pub fn load(path: &Path) -> Result<ParamDictionary, String> {
        let raw = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let text = String::from_utf8_lossy(&raw);
        let root = json::parse(&text)?;

        let feature_keys: Vec<String> = match root.get("feature_keys").and_then(|v| v.as_arr()) {
            Some(items) => items
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            None => DEFAULT_FEATURE_KEYS.iter().map(|s| s.to_string()).collect(),
        };

        let mut entries: Vec<Entry> = Vec::new();
        if let Some(items) = root.get("entries").and_then(|v| v.as_arr()) {
            for item in items {
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let mut feats: Vec<(String, f64)> = Vec::new();
                if let Some(fields) = item.get("features").and_then(|v| v.as_obj()) {
                    for (k, v) in fields {
                        if let Some(x) = v.as_f64() {
                            feats.push((k.clone(), x));
                        }
                    }
                }
                let p = match item.get("params") {
                    Some(p) => p,
                    None => return Err(format!("dictionary entry '{}' has no params", name)),
                };
                let params = Params {
                    k1: p.get("k1").and_then(|v| v.as_f64()).unwrap_or(1.2),
                    b: p.get("b").and_then(|v| v.as_f64()).unwrap_or(0.75),
                    k3: p.get("k3").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    delta: p.get("delta").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    idf_type: p
                        .get("idf_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("rsj")
                        .to_string(),
                };
                entries.push(Entry { name, features: feats, params, vector: Vec::new() });
            }
        }

        let mut mean: HashMap<String, f64> = HashMap::new();
        let mut std_map: HashMap<String, f64> = HashMap::new();
        let mut fmin: HashMap<String, f64> = HashMap::new();
        let mut fmax: HashMap<String, f64> = HashMap::new();
        for key in &feature_keys {
            let vals: Vec<f64> = entries
                .iter()
                .filter_map(|e| e.features.iter().find(|(k, _)| k == key).map(|(_, v)| *v))
                .collect();
            if vals.is_empty() {
                continue;
            }
            let m = stats::mean(&vals);
            let s = stats::std_pop(&vals);
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for v in &vals {
                if *v < lo {
                    lo = *v;
                }
                if *v > hi {
                    hi = *v;
                }
            }
            mean.insert(key.clone(), m);
            std_map.insert(key.clone(), if s == 0.0 { 1.0 } else { s });
            fmin.insert(key.clone(), lo);
            fmax.insert(key.clone(), hi);
        }

        let mut dict = ParamDictionary {
            feature_keys,
            entries,
            mean,
            std: std_map,
            fmin,
            fmax,
            k_neighbors: root.get("k_neighbors").and_then(|v| v.as_i64()).unwrap_or(1).max(1) as usize,
            match_threshold: root
                .get("match_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(2.0),
            normalize: root.get("normalize").and_then(|v| v.as_bool()).unwrap_or(true),
            reject_oov: root.get("reject_oov").and_then(|v| v.as_bool()).unwrap_or(true),
            oov_margin: root.get("oov_margin").and_then(|v| v.as_f64()).unwrap_or(0.5),
            oov_min_features: root
                .get("oov_min_features")
                .and_then(|v| v.as_i64())
                .unwrap_or(2)
                .max(0) as usize,
        };
        dict.precompute_vectors();
        Ok(dict)
    }

    fn precompute_vectors(&mut self) {
        let mut computed: Vec<Vec<(String, f64)>> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            computed.push(self.vector_of(&entry.features));
        }
        for (entry, vec) in self.entries.iter_mut().zip(computed.into_iter()) {
            entry.vector = vec;
        }
    }

    fn vector_of(&self, feats: &[(String, f64)]) -> Vec<(String, f64)> {
        let mut out: Vec<(String, f64)> = Vec::with_capacity(self.feature_keys.len());
        for key in &self.feature_keys {
            let mut v = match feats.iter().find(|(k, _)| k == key) {
                Some((_, value)) => *value,
                None => continue,
            };
            if self.normalize {
                if let (Some(&m), Some(&s)) = (self.mean.get(key), self.std.get(key)) {
                    if s != 0.0 {
                        v = (v - m) / s;
                    }
                }
            }
            out.push((key.clone(), v));
        }
        out
    }

    pub fn vector(&self, f: &Features) -> Vec<(String, f64)> {
        self.vector_of(&f.items)
    }

    /// Features outside the training envelope: at least `oov_min_features` keys fall
    /// outside [min - margin*span, max + margin*span].
    pub fn is_oov(&self, f: &Features) -> bool {
        if !self.reject_oov {
            return false;
        }
        let mut n_oov = 0usize;
        for key in &self.feature_keys {
            let v = match f.get(key) {
                Some(v) => v,
                None => continue,
            };
            let (lo, hi) = match (self.fmin.get(key), self.fmax.get(key)) {
                (Some(&a), Some(&b)) => (a, b),
                _ => continue,
            };
            let span = if hi - lo == 0.0 { 1.0 } else { hi - lo };
            if v < lo - self.oov_margin * span || v > hi + self.oov_margin * span {
                n_oov += 1;
            }
        }
        n_oov >= self.oov_min_features
    }

    pub fn lookup(&self, f: &Features) -> Option<DictLookup> {
        if self.is_oov(f) {
            return None;
        }
        let vec = self.vector(f);
        let mut scored: Vec<(f64, usize)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (distance(&vec, &e.vector), i))
            .collect();
        // Ties break on the entry name, like the Python implementation.
        scored.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| self.entries[a.1].name.cmp(&self.entries[b.1].name))
        });
        if scored.is_empty() {
            return None;
        }
        let best = scored[0];
        if best.0 > self.match_threshold {
            return None;
        }
        let k = self.k_neighbors.min(scored.len());
        let neighbors = &scored[..k];

        let mut k1 = 0.0;
        let mut b = 0.0;
        let mut k3 = 0.0;
        let mut delta = 0.0;
        for (_, idx) in neighbors {
            let p = &self.entries[*idx].params;
            k1 += p.k1;
            b += p.b;
            k3 += p.k3;
            delta += p.delta;
        }
        let kf = k as f64;

        // Majority vote on the IDF variant; first-reached maximum wins, matching
        // Python's `max(votes, key=votes.get)` over an insertion-ordered dict.
        let mut votes: Vec<(String, u32)> = Vec::new();
        for (_, idx) in neighbors {
            let t = &self.entries[*idx].params.idf_type;
            match votes.iter_mut().find(|(name, _)| name == t) {
                Some(slot) => slot.1 += 1,
                None => votes.push((t.clone(), 1)),
            }
        }
        let mut best_vote = ("rsj".to_string(), 0u32);
        for (name, count) in &votes {
            if *count > best_vote.1 {
                best_vote = (name.clone(), *count);
            }
        }

        let params = Params {
            k1: k1 / kf,
            b: b / kf,
            k3: k3 / kf,
            delta: delta / kf,
            idf_type: best_vote.0,
        };
        let matches: Vec<String> = neighbors
            .iter()
            .map(|(_, idx)| self.entries[*idx].name.clone())
            .collect();
        Some(DictLookup { params, distance: stats::round_digits(best.0, 4), matches })
    }

    /// Diagnostics: OOD flag, nearest distance, nearest entry name.
    pub fn probe(&self, f: &Features) -> Json {
        let oov = self.is_oov(f);
        let vec = self.vector(f);
        let mut scored: Vec<(f64, String)> = self
            .entries
            .iter()
            .map(|e| (distance(&vec, &e.vector), e.name.clone()))
            .collect();
        scored.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        json::obj(vec![
            ("oov", Json::Bool(oov)),
            (
                "distance",
                match scored.first() {
                    Some((d, _)) => Json::num(stats::round_digits(*d, 4)),
                    None => Json::Null,
                },
            ),
            (
                "match",
                match scored.first() {
                    Some((_, name)) => Json::str(name.clone()),
                    None => Json::Null,
                },
            ),
        ])
    }
}

fn distance(a: &[(String, f64)], b: &[(String, f64)]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (key, va) in a {
        if let Some((_, vb)) = b.iter().find(|(kb, _)| kb == key) {
            let d = va - vb;
            sum += d * d;
            count += 1;
        }
    }
    if count == 0 {
        f64::INFINITY
    } else {
        (sum / count as f64).sqrt()
    }
}

pub enum Prediction {
    Dictionary(DictLookup),
    Heuristic(Params),
}

impl Prediction {
    pub fn params(&self) -> &Params {
        match self {
            Prediction::Dictionary(hit) => &hit.params,
            Prediction::Heuristic(p) => p,
        }
    }

    pub fn source(&self) -> &'static str {
        match self {
            Prediction::Dictionary(_) => "dictionary",
            Prediction::Heuristic(_) => "heuristic",
        }
    }
}

pub fn predict_with_dictionary(
    f: &Features,
    cfg: &Config,
    dict: Option<&ParamDictionary>,
) -> Prediction {
    if let Some(d) = dict {
        if let Some(hit) = d.lookup(f) {
            return Prediction::Dictionary(hit);
        }
    }
    Prediction::Heuristic(predict(f, cfg))
}
