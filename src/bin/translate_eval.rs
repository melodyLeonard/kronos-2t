//! Chat-style translation probe for `POST /translate`: sends ten EN sentences (FR/ES/DE/YO, each
//! with one to three context lines) to a running `kronos_server` and prints a Markdown table
//! with the output and latency of each, plus the decode rate the server reports.
//!
//! `--latency` instead sends, per case, three short variants of the text with the same context
//! and tone (as the keyboard does while the user types) and prints p50/p95 of the first request
//! (fresh prefix) and of the follow-ups (prefix reused).
//!
//! Usage: `translate_eval [--url http://127.0.0.1:18085] [--label name] [--repeat 1] [--latency]`

use kronos_2t::util::json::Json;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

struct Case {
    text: &'static str,
    to: &'static str,
    tone: &'static str,
    context: &'static [&'static str],
}

const CASES: &[Case] = &[
    Case { text: "see you tomorrow", to: "Spanish", tone: "casual", context: &["Are you coming to Ana's party?", "It starts at 8"] },
    Case { text: "I'm running late, start without me", to: "French", tone: "casual", context: &["Dinner at Marco's tonight?", "I'll bring dessert"] },
    Case { text: "Could you send me the invoice by Friday?", to: "German", tone: "formal", context: &["Thank you for the meeting yesterday, Mr. Weber."] },
    Case { text: "lol no way, he actually said that??", to: "French", tone: "casual", context: &["Did you hear what Tom said in the meeting?", "about the budget"] },
    Case { text: "Happy birthday! Hope you have a great day 🎉", to: "Yoruba", tone: "casual", context: &["Is it your birthday today?"] },
    Case { text: "We regret to inform you that the position has been filled.", to: "Spanish", tone: "formal", context: &["Thank you for applying to the analyst role.", "We reviewed your application carefully."] },
    Case { text: "Can you pick up the kids? I'm stuck at work", to: "German", tone: "casual", context: &["Hey love, how's your day going?", "Mine is chaos"] },
    Case { text: "The package arrived broken, I'd like a refund.", to: "French", tone: "formal", context: &["Order #48213, placed on Monday."] },
    Case { text: "Good morning! Did you sleep well?", to: "Yoruba", tone: "casual", context: &["Good night, talk tomorrow", "Sleep well"] },
    Case { text: "Please find attached the revised contract for your review.", to: "Spanish", tone: "formal", context: &["Following up on our call this morning.", "Legal has approved the changes.", "Let me know if you have questions."] },
];

/// Short (5 to 15 word) follow-up texts sent with each case's context in `--latency` mode.
const VARIANTS: [&str; 3] = [
    "ok see you there at eight",
    "sorry I can't make it tonight, maybe next week?",
    "thanks a lot, that really helps me out",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned());
    let url = flag("--url").unwrap_or_else(|| "http://127.0.0.1:18085".to_string());
    let label = flag("--label").unwrap_or_else(|| url.clone());
    let repeat: usize = flag("--repeat").map_or(1, |v| v.parse().expect("numeric --repeat"));
    let (host, port) = parse_url(&url);

    let status = http(&host, port, "GET", "/v1/admin/status", "").and_then(|b| Json::parse(&b).ok());
    let model = status.as_ref().and_then(|j| j.get("model").and_then(Json::as_str).map(String::from)).unwrap_or_else(|| "?".to_string());
    let tmodel = status.as_ref().and_then(|j| j.get("translate_model").and_then(Json::as_str).map(String::from));
    println!("### {label}\n\nmodel: `{model}`{}\n", tmodel.map_or(String::new(), |t| format!(" · translate model: `{t}`")));
    if args.iter().any(|a| a == "--latency") {
        return latency(&host, port);
    }
    println!("| # | to | tone | input | output | latency |");
    println!("|---|---|---|---|---|---|");
    let mut total = Duration::ZERO;
    let mut out_chars = 0usize;
    for (i, c) in CASES.iter().enumerate() {
        let ctx: Vec<Json> = c.context.iter().map(|s| Json::Str((*s).to_string())).collect();
        let body = Json::Obj(vec![
            ("text".into(), Json::Str(c.text.into())),
            ("to".into(), Json::Str(c.to.into())),
            ("tone".into(), Json::Str(c.tone.into())),
            ("context".into(), Json::Arr(ctx)),
        ])
        .stringify();
        let mut best = Duration::MAX;
        let mut text = String::new();
        for _ in 0..repeat.max(1) {
            let t0 = Instant::now();
            let resp = http(&host, port, "POST", "/translate", &body).unwrap_or_default();
            let dt = t0.elapsed();
            text = Json::parse(&resp)
                .ok()
                .and_then(|j| j.get("text").and_then(Json::as_str).map(String::from))
                .unwrap_or_else(|| format!("<error: {}>", resp.trim()));
            best = best.min(dt);
        }
        total += best;
        out_chars += text.chars().count();
        println!(
            "| {} | {} | {} | {} | {} | {:.2}s |",
            i + 1,
            c.to,
            c.tone,
            cell(c.text),
            cell(&text),
            best.as_secs_f64()
        );
    }
    println!(
        "\ntotal {:.2}s over {} requests, {:.0} output chars/s\n",
        total.as_secs_f64(),
        CASES.len(),
        out_chars as f64 / total.as_secs_f64().max(1e-9)
    );
}

fn request(host: &str, port: u16, c: &Case, text: &str) -> (String, Duration) {
    let ctx: Vec<Json> = c.context.iter().map(|s| Json::Str((*s).to_string())).collect();
    let body = Json::Obj(vec![
        ("text".into(), Json::Str(text.into())),
        ("to".into(), Json::Str(c.to.into())),
        ("tone".into(), Json::Str(c.tone.into())),
        ("context".into(), Json::Arr(ctx)),
    ])
    .stringify();
    let t0 = Instant::now();
    let resp = http(host, port, "POST", "/translate", &body).unwrap_or_default();
    let dt = t0.elapsed();
    let text = Json::parse(&resp)
        .ok()
        .and_then(|j| j.get("text").and_then(Json::as_str).map(String::from))
        .unwrap_or_else(|| format!("<error: {}>", resp.trim()));
    (text, dt)
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

/// `--latency`: first request per case (new context) vs the follow-ups sharing it.
fn latency(host: &str, port: u16) {
    let (mut first, mut warm) = (Vec::new(), Vec::new());
    println!("| # | to | first (new context) | follow-ups (same context) |");
    println!("|---|---|---|---|");
    for (i, c) in CASES.iter().enumerate() {
        let (_, d0) = request(host, port, c, c.text);
        let mut ds = Vec::new();
        for v in VARIANTS {
            let (out, d) = request(host, port, c, v);
            assert!(!out.starts_with("<error"), "{out}");
            ds.push(d.as_secs_f64());
        }
        first.push(d0.as_secs_f64());
        warm.extend(ds.iter().copied());
        let cells: Vec<String> = ds.iter().map(|d| format!("{:.0} ms", d * 1e3)).collect();
        println!("| {} | {} | {:.0} ms | {} |", i + 1, c.to, d0.as_secs_f64() * 1e3, cells.join(" / "));
    }
    println!(
        "\nfirst request: p50 {:.0} ms, p95 {:.0} ms · follow-ups with the same context: p50 {:.0} ms, p95 {:.0} ms (n={})\n",
        pct(&mut first, 0.5) * 1e3,
        pct(&mut first, 0.95) * 1e3,
        pct(&mut warm, 0.5) * 1e3,
        pct(&mut warm, 0.95) * 1e3,
        warm.len()
    );
}

fn cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', "<br>")
}

fn parse_url(url: &str) -> (String, u16) {
    let rest = url.trim_start_matches("http://").trim_end_matches('/');
    match rest.split_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().expect("port")),
        None => (rest.to_string(), 80),
    }
}

fn http(host: &str, port: u16, method: &str, path: &str, body: &str) -> Option<String> {
    let mut s = TcpStream::connect((host, port)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(600))).ok()?;
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .ok()?;
    let mut out = String::new();
    s.read_to_string(&mut out).ok()?;
    out.split_once("\r\n\r\n").map(|x| x.1.to_string())
}
