//! FLORES-200 devtest accuracy of the translation prompt, English → each keyboard language.
//! Scores chrF++ (char 1–6 + word 1–2 n-grams, β=2) and corpus BLEU-4 against the references and
//! writes a Markdown report plus one TSV of `idx \t source \t reference \t hypothesis` per language.
//!
//! Usage: `translate_bench <model_dir> --flores <flores200_dataset dir> [--mode nf4g128] [--n 1012]
//!        [--langs FR,ES,...] [--threads N] [--cpu] [--out DIR] [--split devtest|dev]`
//! Languages already present as a complete TSV in `--out` are skipped, so a long run can resume.

use kronos_2t::runtime::chat::ChatModel;
use kronos_2t::runtime::translate::{clean_translation, run, TranslateRequest};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Keyboard code, English name used in the prompt, FLORES-200 file stem.
const LANGS: &[(&str, &str, &str)] = &[
    ("FR", "French", "fra_Latn"),
    ("ES", "Spanish", "spa_Latn"),
    ("PT", "Portuguese", "por_Latn"),
    ("DE", "German", "deu_Latn"),
    ("IT", "Italian", "ita_Latn"),
    ("NL", "Dutch", "nld_Latn"),
    ("PL", "Polish", "pol_Latn"),
    ("CS", "Czech", "ces_Latn"),
    ("SK", "Slovak", "slk_Latn"),
    ("HU", "Hungarian", "hun_Latn"),
    ("RO", "Romanian", "ron_Latn"),
    ("SV", "Swedish", "swe_Latn"),
    ("DA", "Danish", "dan_Latn"),
    ("NO", "Norwegian", "nob_Latn"),
    ("FI", "Finnish", "fin_Latn"),
    ("EL", "Greek", "ell_Grek"),
    ("UK", "Ukrainian", "ukr_Cyrl"),
    ("RU", "Russian", "rus_Cyrl"),
    ("KK", "Kazakh", "kaz_Cyrl"),
    ("MN", "Mongolian", "khk_Cyrl"),
    ("TR", "Turkish", "tur_Latn"),
    ("AR", "Arabic", "arb_Arab"),
    ("FA", "Persian", "pes_Arab"),
    ("UR", "Urdu", "urd_Arab"),
    ("UG", "Uyghur", "uig_Arab"),
    ("HE", "Hebrew", "heb_Hebr"),
    ("HI", "Hindi", "hin_Deva"),
    ("BN", "Bengali", "ben_Beng"),
    ("GU", "Gujarati", "guj_Gujr"),
    ("MR", "Marathi", "mar_Deva"),
    ("TA", "Tamil", "tam_Taml"),
    ("TE", "Telugu", "tel_Telu"),
    ("BO", "Tibetan", "bod_Tibt"),
    ("ZH", "Chinese (Simplified)", "zho_Hans"),
    ("ZT", "Chinese (Traditional)", "zho_Hant"),
    ("YUE", "Cantonese", "yue_Hant"),
    ("JA", "Japanese", "jpn_Jpan"),
    ("KO", "Korean", "kor_Hang"),
    ("VI", "Vietnamese", "vie_Latn"),
    ("TH", "Thai", "tha_Thai"),
    ("KM", "Khmer", "khm_Khmr"),
    ("MY", "Burmese", "mya_Mymr"),
    ("ID", "Indonesian", "ind_Latn"),
    ("MS", "Malay", "zsm_Latn"),
    ("FIL", "Filipino", "tgl_Latn"),
    ("YO", "Yoruba", "yor_Latn"),
    ("IG", "Igbo", "ibo_Latn"),
    ("HA", "Hausa", "hau_Latn"),
    ("SW", "Swahili", "swh_Latn"),
    ("AM", "Amharic", "amh_Ethi"),
    ("ZU", "Zulu", "zul_Latn"),
];

/// Scripts without word spaces: BLEU and chrF word n-grams use characters as words.
const CHAR_TOKENIZED: &[&str] = &["zho_Hans", "zho_Hant", "yue_Hant", "jpn_Jpan", "tha_Thai", "khm_Khmr", "mya_Mymr", "bod_Tibt"];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.iter().any(|a| a == "--help") {
        eprintln!("usage: translate_bench <model_dir> --flores DIR [--mode nf4g128] [--n 1012] [--langs FR,ES] [--threads N] [--cpu] [--out DIR]");
        std::process::exit(2);
    }
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned());
    let model_dir = PathBuf::from(&args[1]);
    let flores = PathBuf::from(flag("--flores").expect("--flores DIR"));
    let mode = flag("--mode").unwrap_or_else(|| "nf4g128".into());
    let n: usize = flag("--n").map_or(1012, |v| v.parse().expect("--n"));
    let threads: usize = flag("--threads").map_or(0, |v| v.parse().expect("--threads"));
    let out = PathBuf::from(flag("--out").unwrap_or_else(|| "translate_bench_out".into()));
    let split = flag("--split").unwrap_or_else(|| "devtest".into());
    let langs: Vec<&(&str, &str, &str)> = match flag("--langs") {
        Some(l) => l.split(',').map(|c| LANGS.iter().find(|x| x.0.eq_ignore_ascii_case(c.trim())).unwrap_or_else(|| panic!("unknown lang {c}"))).collect(),
        None => LANGS.iter().collect(),
    };
    fs::create_dir_all(&out).expect("out dir");

    let src = read_lines(&flores.join(&split).join(format!("eng_Latn.{split}")));
    let n = n.min(src.len());

    let threads = if threads == 0 { kronos_2t::kernels::thread_pool::default_inference_threads() } else { threads };
    let mut model = ChatModel::load(&model_dir, &mode, 2048, threads, None).expect("load model");
    if !args.iter().any(|a| a == "--cpu") {
        if let Err(e) = model.enable_metal() {
            eprintln!("metal unavailable ({e}); running on CPU");
        }
    }
    eprintln!("{} · {} · {} sentences × {} languages", model.name, mode, n, langs.len());

    let report = out.join("report.md");
    let mut rows: Vec<String> = Vec::new();
    for (code, name, stem) in langs {
        let refs = read_lines(&flores.join(&split).join(format!("{stem}.{split}")));
        let tsv = out.join(format!("{code}.tsv"));
        let char_tok = CHAR_TOKENIZED.contains(stem);
        let (hyps, secs, toks) = if let Some(h) = load_complete_tsv(&tsv, n) {
            (h, 0.0, 0)
        } else {
            let mut f = fs::File::create(&tsv).expect("tsv");
            let mut hyps = Vec::with_capacity(n);
            let (mut secs, mut toks) = (0.0f64, 0usize);
            let t_lang = Instant::now();
            for i in 0..n {
                let req = TranslateRequest { text: &src[i], to: name, from: "English", tone: "neutral", context: &[], temperature: 0.0 };
                let t0 = Instant::now();
                let hyp = match run(&mut model, &req, |_| true) {
                    Ok(c) => {
                        toks += c.completion_tokens;
                        clean_translation(&c.text)
                    }
                    Err(e) => {
                        eprintln!("{code} #{i}: {e}");
                        String::new()
                    }
                };
                secs += t0.elapsed().as_secs_f64();
                writeln!(f, "{i}\t{}\t{}\t{}", tab_safe(&src[i]), tab_safe(&refs[i]), tab_safe(&hyp)).expect("write");
                hyps.push(hyp);
                if (i + 1) % 100 == 0 {
                    eprintln!("{code}: {}/{n} · {:.0} s", i + 1, t_lang.elapsed().as_secs_f64());
                }
            }
            (hyps, secs, toks)
        };
        let chrf = chrf_pp(&hyps, &refs[..n], char_tok);
        let bleu = corpus_bleu(&hyps, &refs[..n], char_tok);
        let speed = if secs > 0.0 { format!("{:.2} s · {:.1} tok/s", secs / n as f64, toks as f64 / secs) } else { "cached".into() };
        let row = format!("| {code} | {name} | {n} | {chrf:.1} | {bleu:.1} | {speed} |");
        eprintln!("{row}");
        rows.push(row);
        write_report(&report, &model.name, &mode, &rows);
    }
}

fn write_report(path: &Path, model: &str, mode: &str, rows: &[String]) {
    let mut s = format!("# FLORES-200, English → X\n\nmodel: `{model}` · mode `{mode}` · greedy · no context\n\n");
    s.push_str("| code | language | n | chrF++ | BLEU | per sentence |\n|---|---|---|---|---|---|\n");
    for r in rows {
        s.push_str(r);
        s.push('\n');
    }
    fs::write(path, s).expect("report");
}

fn read_lines(p: &Path) -> Vec<String> {
    fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).lines().map(str::to_string).collect()
}

fn tab_safe(s: &str) -> String {
    s.replace(['\t', '\n'], " ")
}

/// Hypotheses from an earlier run, if the TSV has all `n` rows.
fn load_complete_tsv(p: &Path, n: usize) -> Option<Vec<String>> {
    let text = fs::read_to_string(p).ok()?;
    let hyps: Vec<String> = text.lines().filter_map(|l| l.split('\t').nth(3).map(str::to_string)).collect();
    (hyps.len() == n).then_some(hyps)
}

fn words(s: &str, char_tok: bool) -> Vec<String> {
    if char_tok {
        return s.chars().filter(|c| !c.is_whitespace()).map(|c| c.to_string()).collect();
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() || c == '\'' {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if !c.is_whitespace() {
                out.push(c.to_string());
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn ngram_counts<T: std::hash::Hash + Eq + Clone>(items: &[T], n: usize) -> HashMap<Vec<T>, usize> {
    let mut m = HashMap::new();
    if items.len() >= n {
        for w in items.windows(n) {
            *m.entry(w.to_vec()).or_insert(0) += 1;
        }
    }
    m
}

/// Matches, hypothesis total and reference total for order `n` over `items`.
fn ngram_stats<T: std::hash::Hash + Eq + Clone>(hyp: &[T], refr: &[T], n: usize) -> (usize, usize, usize) {
    let h = ngram_counts(hyp, n);
    let r = ngram_counts(refr, n);
    let matches = h.iter().map(|(g, &c)| c.min(*r.get(g).unwrap_or(&0))).sum();
    (matches, h.values().sum(), r.values().sum())
}

/// chrF++ as in sacrebleu: character n-grams 1–6 (whitespace removed) plus word n-grams 1–2,
/// β = 2, precision/recall summed over the corpus per order and averaged over orders.
fn chrf_pp(hyps: &[String], refs: &[String], char_tok: bool) -> f64 {
    const BETA2: f64 = 4.0;
    let mut stats: Vec<(usize, usize, usize)> = vec![(0, 0, 0); 8];
    for (h, r) in hyps.iter().zip(refs) {
        let hc: Vec<char> = h.chars().filter(|c| !c.is_whitespace()).collect();
        let rc: Vec<char> = r.chars().filter(|c| !c.is_whitespace()).collect();
        for n in 1..=6 {
            let (m, ht, rt) = ngram_stats(&hc, &rc, n);
            stats[n - 1].0 += m;
            stats[n - 1].1 += ht;
            stats[n - 1].2 += rt;
        }
        let hw = words(h, char_tok);
        let rw = words(r, char_tok);
        for n in 1..=2 {
            let (m, ht, rt) = ngram_stats(&hw, &rw, n);
            stats[5 + n].0 += m;
            stats[5 + n].1 += ht;
            stats[5 + n].2 += rt;
        }
    }
    let mut f_sum = 0.0;
    let mut orders = 0;
    for &(m, ht, rt) in &stats {
        if rt == 0 {
            continue;
        }
        orders += 1;
        let p = if ht > 0 { m as f64 / ht as f64 } else { 0.0 };
        let r = m as f64 / rt as f64;
        if p + r > 0.0 {
            f_sum += (1.0 + BETA2) * p * r / (BETA2 * p + r);
        }
    }
    if orders == 0 { 0.0 } else { 100.0 * f_sum / orders as f64 }
}

/// Corpus BLEU-4 with brevity penalty over the simple tokenization in [`words`].
fn corpus_bleu(hyps: &[String], refs: &[String], char_tok: bool) -> f64 {
    let mut matches = [0usize; 4];
    let mut totals = [0usize; 4];
    let (mut hyp_len, mut ref_len) = (0usize, 0usize);
    for (h, r) in hyps.iter().zip(refs) {
        let hw = words(h, char_tok);
        let rw = words(r, char_tok);
        hyp_len += hw.len();
        ref_len += rw.len();
        for n in 1..=4 {
            let (m, ht, _) = ngram_stats(&hw, &rw, n);
            matches[n - 1] += m;
            totals[n - 1] += ht;
        }
    }
    if totals.iter().any(|&t| t == 0) || matches[0] == 0 {
        return 0.0;
    }
    let log_p: f64 = (0..4).map(|i| ((matches[i].max(1)) as f64 / totals[i] as f64).ln()).sum::<f64>() / 4.0;
    let bp = if hyp_len >= ref_len { 1.0 } else { (1.0 - ref_len as f64 / hyp_len as f64).exp() };
    100.0 * bp * log_p.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_match_scores_100() {
        let s = vec!["The cat sat on the mat.".to_string(), "Hello, world!".to_string()];
        assert!((chrf_pp(&s, &s, false) - 100.0).abs() < 1e-9);
        assert!((corpus_bleu(&s, &s, false) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn mismatch_scores_low_and_char_tokenization_splits() {
        let h = vec!["completely different text here".to_string()];
        let r = vec!["the quick brown fox jumps".to_string()];
        assert!(chrf_pp(&h, &r, false) < 15.0);
        assert_eq!(corpus_bleu(&h, &r, false), 0.0);
        assert_eq!(words("今天 天气", true).len(), 4);
    }
}
