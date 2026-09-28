//! Minimal JSON parser and writer (std only, no external crates).
//!
//! Two usage modes:
//! * `parse` / `Json` tree - used for the small config-style files
//!   (param_dictionary.json).
//! * `scan_object` - streams a JSON object line and only materializes the keys the
//!   caller asks for, skipping everything else without allocating. This keeps the
//!   multi-GB corpus files cheap to read.
//!
//! Output mirrors `json.dump(..., indent=2)` / `json.dumps(...)` from Python so the
//! two implementations produce diffable files.

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

/// Build an object literal, keeping the field order given.
pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
    Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// Format a float the way Python's `repr()`/`str()` does for a float.
pub fn py_float_str(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() };
    }
    let s = format!("{}", v);
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        s + ".0"
    }
}

impl Json {
    pub fn str<S: Into<String>>(s: S) -> Json {
        Json::Str(s.into())
    }

    pub fn num(v: f64) -> Json {
        Json::Num(v)
    }

    pub fn int(v: i64) -> Json {
        Json::Int(v)
    }

    pub fn arr(v: Vec<Json>) -> Json {
        Json::Arr(v)
    }

    /// Object lookup by key (linear scan; our objects are small).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            Json::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Int(v) => Some(*v),
            Json::Num(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Identifier-ish value as text, matching Python's `str(value)`.
    pub fn to_id(&self) -> Option<String> {
        match self {
            Json::Str(s) => Some(s.clone()),
            Json::Int(v) => Some(v.to_string()),
            Json::Num(v) => Some(py_float_str(*v)),
            Json::Bool(b) => Some(if *b { "True".to_string() } else { "False".to_string() }),
            _ => None,
        }
    }

    /// Pretty output, 2-space indent, like `json.dump(v, f, ensure_ascii=False, indent=2)`.
    pub fn dump_pretty(&self) -> String {
        let mut out = String::new();
        self.write_pretty(0, &mut out);
        out
    }

    /// Compact output with Python's default separators (", " and ": ").
    pub fn dump(&self) -> String {
        let mut out = String::new();
        self.write_compact(&mut out);
        out
    }

    fn write_pretty(&self, indent: usize, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(v) => {
                let _ = write!(out, "{}", v);
            }
            Json::Num(v) => out.push_str(&py_float_str(*v)),
            Json::Str(s) => write_py_string(s, out),
            Json::Arr(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                for (i, item) in items.iter().enumerate() {
                    pad(indent + 2, out);
                    item.write_pretty(indent + 2, out);
                    if i + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                pad(indent, out);
                out.push(']');
            }
            Json::Obj(fields) => {
                if fields.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                for (i, (k, v)) in fields.iter().enumerate() {
                    pad(indent + 2, out);
                    write_py_string(k, out);
                    out.push_str(": ");
                    v.write_pretty(indent + 2, out);
                    if i + 1 < fields.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                pad(indent, out);
                out.push('}');
            }
        }
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(v) => {
                let _ = write!(out, "{}", v);
            }
            Json::Num(v) => out.push_str(&py_float_str(*v)),
            Json::Str(s) => write_py_string(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    item.write_compact(out);
                }
                out.push(']');
            }
            Json::Obj(fields) => {
                out.push('{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_py_string(k, out);
                    out.push_str(": ");
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }
}

fn pad(n: usize, out: &mut String) {
    for _ in 0..n {
        out.push(' ');
    }
}

fn write_py_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ---------------------------------------------------------------- parsing

pub fn parse(input: &str) -> Result<Json, String> {
    let mut p = Parser { b: input.as_bytes(), i: 0 };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    Ok(v)
}

/// Parse the top-level object of `line` and return only `want` keys.
///
/// Unknown keys (for example a big `metadata` blob) are skipped without allocating.
pub fn scan_object(line: &str, want: &[&str], out: &mut Vec<(String, Json)>) -> Result<(), String> {
    let mut p = Parser { b: line.as_bytes(), i: 0 };
    p.skip_ws();
    if p.peek() != Some(b'{') {
        return Err("expected a JSON object".to_string());
    }
    p.i += 1;
    p.skip_ws();
    if p.peek() == Some(b'}') {
        return Ok(());
    }
    loop {
        p.skip_ws();
        if p.peek() != Some(b'"') {
            return Err(format!("expected object key at byte {}", p.i));
        }
        let key = p.string()?;
        p.skip_ws();
        if p.peek() != Some(b':') {
            return Err(format!("expected ':' at byte {}", p.i));
        }
        p.i += 1;
        if want.contains(&key.as_str()) {
            let v = p.value()?;
            out.push((key, v));
        } else {
            p.skip_value()?;
        }
        p.skip_ws();
        match p.peek() {
            Some(b',') => p.i += 1,
            Some(b'}') => break,
            _ => return Err(format!("expected ',' or '}}' at byte {}", p.i)),
        }
    }
    Ok(())
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            None => Err("unexpected end of input".to_string()),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => {
                self.lit("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.lit("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.lit("null")?;
                Ok(Json::Null)
            }
            Some(_) => self.number(),
        }
    }

    fn lit(&mut self, text: &str) -> Result<(), String> {
        let bytes = text.as_bytes();
        if self.i + bytes.len() <= self.b.len() && &self.b[self.i..self.i + bytes.len()] == bytes {
            self.i += bytes.len();
            Ok(())
        } else {
            Err(format!("invalid literal at byte {}", self.i))
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.i += 1;
        let mut fields: Vec<(String, Json)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(fields));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected object key at byte {}", self.i));
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at byte {}", self.i));
            }
            self.i += 1;
            let v = self.value()?;
            fields.push((key, v));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
            }
        }
        Ok(Json::Obj(fields))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.i += 1;
        let mut items: Vec<Json> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let v = self.value()?;
            items.push(v);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
            }
        }
        Ok(Json::Arr(items))
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            let c = match self.peek() {
                Some(c) => c,
                None => return Err("unterminated string".to_string()),
            };
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    let e = match self.peek() {
                        Some(e) => e,
                        None => return Err("unterminated escape".to_string()),
                    };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            if (0xD800..0xDC00).contains(&hi) {
                                let mut combined: Option<char> = None;
                                if self.peek() == Some(b'\\') && self.b.get(self.i + 1) == Some(&b'u') {
                                    self.i += 2;
                                    let lo = self.hex4()?;
                                    if (0xDC00..0xE000).contains(&lo) {
                                        let cp = 0x10000u32 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                                        combined = char::from_u32(cp);
                                    }
                                }
                                out.push(combined.unwrap_or('\u{FFFD}'));
                            } else {
                                out.push(char::from_u32(hi).unwrap_or('\u{FFFD}'));
                            }
                        }
                        other => out.push(other as char),
                    }
                }
                _ => {
                    let len = utf8_len(c);
                    if self.i + len > self.b.len() {
                        return Err("truncated utf-8 sequence".to_string());
                    }
                    let s = std::str::from_utf8(&self.b[self.i..self.i + len])
                        .map_err(|e| e.to_string())?;
                    out.push_str(s);
                    self.i += len;
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("truncated \\u escape".to_string());
        }
        let mut v: u32 = 0;
        for k in 0..4 {
            let c = self.b[self.i + k];
            let d = match c {
                b'0'..=b'9' => (c - b'0') as u32,
                b'a'..=b'f' => (c - b'a') as u32 + 10,
                b'A'..=b'F' => (c - b'A') as u32 + 10,
                _ => return Err("invalid hex digit in \\u escape".to_string()),
            };
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E' => self.i += 1,
                _ => break,
            }
        }
        if self.i == start {
            return Err(format!("invalid number at byte {}", start));
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
        if s.contains('.') || s.contains('e') || s.contains('E') {
            match s.parse::<f64>() {
                Ok(v) => Ok(Json::Num(v)),
                Err(e) => Err(e.to_string()),
            }
        } else {
            match s.parse::<i64>() {
                Ok(v) => Ok(Json::Int(v)),
                Err(_) => match s.parse::<f64>() {
                    Ok(v) => Ok(Json::Num(v)),
                    Err(e) => Err(e.to_string()),
                },
            }
        }
    }

    /// Walk one value without building anything.
    fn skip_value(&mut self) -> Result<(), String> {
        self.skip_ws();
        match self.peek() {
            None => Err("unexpected end of input".to_string()),
            Some(b'{') => {
                self.i += 1;
                self.skip_ws();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    self.skip_ws();
                    if self.peek() != Some(b'"') {
                        return Err(format!("expected object key at byte {}", self.i));
                    }
                    self.string()?;
                    self.skip_ws();
                    if self.peek() != Some(b':') {
                        return Err(format!("expected ':' at byte {}", self.i));
                    }
                    self.i += 1;
                    self.skip_value()?;
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(());
                        }
                        _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                self.skip_ws();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    self.skip_value()?;
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(());
                        }
                        _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
                    }
                }
            }
            Some(b'"') => {
                self.string()?;
                Ok(())
            }
            Some(b't') => self.lit("true"),
            Some(b'f') => self.lit("false"),
            Some(b'n') => self.lit("null"),
            Some(_) => {
                let start = self.i;
                while let Some(c) = self.peek() {
                    match c {
                        b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E' => self.i += 1,
                        _ => break,
                    }
                }
                if self.i == start {
                    Err(format!("invalid JSON value at byte {}", start))
                } else {
                    Ok(())
                }
            }
        }
    }
}

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1
    }
}
