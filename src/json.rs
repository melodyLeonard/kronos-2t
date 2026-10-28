//! Minimal JSON values for the HTTP API.

use crate::safetensors::JsonCursor;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut c = JsonCursor::new(text);
        let v = parse_value(&mut c, 0)?;
        if c.peek().is_some() {
            return Err("trailing characters after JSON value".to_string());
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
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
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(kv) => Some(kv),
            _ => None,
        }
    }

    /// Serializes this value into compact canonical JSON text.
    pub fn stringify(&self) -> String {
        let mut out = String::new();
        self.write_into(&mut out);
        out
    }

    fn write_into(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if !n.is_finite() {
                    out.push_str("null");
                } else if n.fract() == 0.0 && *n >= (i64::MIN as f64) && *n <= (i64::MAX as f64) {
                    out.push_str(&format!("{}", *n as i64));
                } else {
                    out.push_str(&format!("{n}"));
                }
            }
            Json::Str(s) => out.push_str(&quote(s)),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_into(out);
                }
                out.push(']');
            }
            Json::Obj(kv) => {
                out.push('{');
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&quote(k));
                    out.push(':');
                    v.write_into(out);
                }
                out.push('}');
            }
        }
    }
}

const MAX_DEPTH: usize = 64;

fn parse_value(c: &mut JsonCursor, depth: usize) -> Result<Json, String> {
    if depth > MAX_DEPTH {
        return Err("JSON nested too deeply".to_string());
    }
    match c.peek().ok_or("unexpected end of JSON")? {
        b'"' => Ok(Json::Str(c.parse_string()?)),
        b'{' => {
            c.expect_byte(b'{')?;
            let mut kv = Vec::new();
            if c.peek() == Some(b'}') {
                c.expect_byte(b'}')?;
                return Ok(Json::Obj(kv));
            }
            loop {
                let k = c.parse_string()?;
                c.expect_byte(b':')?;
                kv.push((k, parse_value(c, depth + 1)?));
                match c.peek() {
                    Some(b',') => c.expect_byte(b',')?,
                    _ => {
                        c.expect_byte(b'}')?;
                        return Ok(Json::Obj(kv));
                    }
                }
            }
        }
        b'[' => {
            c.expect_byte(b'[')?;
            let mut items = Vec::new();
            if c.peek() == Some(b']') {
                c.expect_byte(b']')?;
                return Ok(Json::Arr(items));
            }
            loop {
                items.push(parse_value(c, depth + 1)?);
                match c.peek() {
                    Some(b',') => c.expect_byte(b',')?,
                    _ => {
                        c.expect_byte(b']')?;
                        return Ok(Json::Arr(items));
                    }
                }
            }
        }
        b't' => c.consume_literal(b"true").map(|_| Json::Bool(true)),
        b'f' => c.consume_literal(b"false").map(|_| Json::Bool(false)),
        b'n' => c.consume_literal(b"null").map(|_| Json::Null),
        _ => Ok(Json::Num(c.parse_f64()?)),
    }
}

/// `s` as a quoted JSON string.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_quote() {
        let v = Json::parse(r#"{"a": [1, 2.5, -3e2], "b": {"c": "x\"yé"}, "d": true, "e": null}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_arr().unwrap()[2].as_f64(), Some(-300.0));
        assert_eq!(v.get("b").unwrap().get("c").unwrap().as_str(), Some("x\"yé"));
        assert_eq!(v.get("d").unwrap().as_bool(), Some(true));
        assert_eq!(v.get("e"), Some(&Json::Null));
        assert!(Json::parse("{\"a\":1} x").is_err());
        let s = "line\n\"q\"\t\u{1}";
        assert_eq!(Json::parse(&quote(s)).unwrap().as_str(), Some(s));
    }
}
