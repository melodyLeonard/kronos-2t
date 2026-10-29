//! Local on-device speech-to-text pipeline backed by macOS SpeechAnalyzer on the Apple Neural Engine.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::device::{global_device_manager, split_camel_case_parts};
use crate::json::quote;
use crate::vision::decode_base64;

#[derive(Clone, Debug)]
pub struct VoiceTranscription {
    pub raw_text: String,
    pub text: String,
    pub audio_ms: u64,
    pub transcribe_ms: f64,
    pub engine: &'static str,
}

#[derive(Clone, Debug)]
pub struct PreparedWav {
    pub wav_bytes: Vec<u8>,
    pub audio_ms: u64,
    pub speech_detected: bool,
}

static SPEECH_WARMED: AtomicBool = AtomicBool::new(false);
static SPEECH_REQ_SEQ: AtomicU64 = AtomicU64::new(1);

// SFSpeechRecognizer triggers a TCC crash when the parent host bundle lacks NSSpeechRecognitionUsageDescription;
// SpeechAnalyzer + SpeechTranscriber runs offline on the Neural Engine without that TCC gate.
#[cfg(target_os = "macos")]
const SWIFT_SPEECH_WORKER_SRC: &str = r#"import Foundation
import Speech
import AVFoundation

@main
struct KronosSpeechWorker {
    static func emit(_ line: String) {
        if let data = (line + "\n").data(using: .utf8) {
            FileHandle.standardOutput.write(data)
        }
    }

    static func sanitize(_ s: String) -> String {
        s.replacingOccurrences(of: "\r", with: " ")
         .replacingOccurrences(of: "\n", with: " ")
         .replacingOccurrences(of: "\t", with: " ")
         .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func resolveLocale() async -> Locale {
        let preferred = Locale(identifier: "en_US")
        if let matched = await SpeechTranscriber.supportedLocale(equivalentTo: preferred) {
            return matched
        }
        if let cur = await SpeechTranscriber.supportedLocale(equivalentTo: Locale.current) {
            return cur
        }
        return preferred
    }

    static func transcribeFile(path: String, hints: [String], locale: Locale) async throws -> String {
        let url = URL(fileURLWithPath: path)
        let audioFile = try AVAudioFile(forReading: url)
        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
        let options = SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .processLifetime)
        let analyzer = SpeechAnalyzer(modules: [transcriber], options: options)
        if !hints.isEmpty {
            let ctx = AnalysisContext()
            ctx.contextualStrings[.general] = hints
            try await analyzer.setContext(ctx)
        }
        async let collected: String = {
            var pieces: [String] = []
            for try await res in transcriber.results {
                if res.isFinal {
                    let seg = String(res.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !seg.isEmpty {
                        pieces.append(seg)
                    }
                }
            }
            return pieces.joined(separator: " ")
        }()
        if let lastSample = try await analyzer.analyzeSequence(from: audioFile) {
            try await analyzer.finalizeAndFinish(through: lastSample)
        } else {
            await analyzer.cancelAndFinishNow()
        }
        return try await collected
    }

    static func main() async {
        guard SpeechTranscriber.isAvailable else {
            emit("ERR\tSpeechTranscriber unavailable on this Mac")
            return
        }
        let locale = await resolveLocale()
        while let rawLine = readLine(strippingNewline: true) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }
            if line == "QUIT" { break }
            if line.hasPrefix("WARM") {
                let t0 = ContinuousClock.now
                let warmTranscriber = SpeechTranscriber(locale: locale, preset: .transcription)
                if let req = try? await AssetInventory.assetInstallationRequest(supporting: [warmTranscriber]) {
                    try? await req.downloadAndInstall()
                }
                let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: true)
                if parts.count > 1 {
                    let warmPath = String(parts[1])
                    _ = try? await transcribeFile(path: warmPath, hints: ["Kronos"], locale: locale)
                } else {
                    let opts = SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .processLifetime)
                    let analyzer = SpeechAnalyzer(modules: [warmTranscriber], options: opts)
                    await analyzer.cancelAndFinishNow()
                }
                let dur = t0.duration(to: ContinuousClock.now)
                let ms = Double(dur.components.seconds) * 1000.0 + Double(dur.components.attoseconds) / 1e15
                emit(String(format: "OK\t%.2f\tREADY", ms))
                continue
            }
            let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            let wavPath = String(parts[0])
            var hints: [String] = []
            if parts.count > 1 {
                hints = parts[1]
                    .split(separator: "|", omittingEmptySubsequences: true)
                    .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
            }
            let t0 = ContinuousClock.now
            do {
                let text = try await transcribeFile(path: wavPath, hints: hints, locale: locale)
                let dur = t0.duration(to: ContinuousClock.now)
                let ms = Double(dur.components.seconds) * 1000.0 + Double(dur.components.attoseconds) / 1e15
                emit(String(format: "OK\t%.2f\t%@", ms, sanitize(text)))
            } catch {
                emit("ERR\t" + sanitize(String(describing: error)))
            }
        }
    }
}
"#;

#[cfg(target_os = "macos")]
struct SpeechWorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[cfg(target_os = "macos")]
impl Drop for SpeechWorkerProcess {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "QUIT");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(target_os = "macos")]
static GLOBAL_SPEECH_WORKER: OnceLock<Mutex<Option<SpeechWorkerProcess>>> = OnceLock::new();

#[cfg(target_os = "macos")]
fn worker_slot() -> &'static Mutex<Option<SpeechWorkerProcess>> {
    GLOBAL_SPEECH_WORKER.get_or_init(|| Mutex::new(None))
}

pub fn prewarm_speech() {
    #[cfg(target_os = "macos")]
    {
        if SPEECH_WARMED.load(Ordering::Relaxed) {
            return;
        }
        let silent_samples = vec![0i16; 3200];
        let warm_wav = encode_pcm16_mono_wav(&silent_samples, 16_000);
        let warm_path = temp_audio_path("warm", "wav");
        let wrote = fs::write(&warm_path, &warm_wav).is_ok();
        let cmd = if wrote {
            format!("WARM\t{}", warm_path.display())
        } else {
            "WARM".to_string()
        };
        let res = send_worker_command(&cmd);
        if wrote {
            let _ = fs::remove_file(&warm_path);
        }
        if res.is_ok() {
            SPEECH_WARMED.store(true, Ordering::Relaxed);
        }
    }
}

pub fn speech_status_json() -> String {
    #[cfg(target_os = "macos")]
    {
        let available = Path::new("/usr/bin/swiftc").exists();
        let warmed = SPEECH_WARMED.load(Ordering::Relaxed);
        format!(
            r#"{{"available":{},"warmed":{},"engine":"macos-speech-analyzer-ane"}}"#,
            available, warmed
        )
    }
    #[cfg(not(target_os = "macos"))]
    {
        r#"{"available":false,"warmed":false,"engine":"unsupported"}"#.to_string()
    }
}

pub fn voice_transcription_to_json(v: &VoiceTranscription) -> String {
    format!(
        r#"{{"text":{},"raw_text":{},"audio_ms":{},"transcribe_ms":{:.2},"engine":{}}}"#,
        quote(&v.text),
        quote(&v.raw_text),
        v.audio_ms,
        v.transcribe_ms,
        quote(v.engine),
    )
}

/// Decodes a `data:audio/...;base64,...` URL or raw Base64 payload into raw container bytes.
pub fn decode_audio_payload(input: &str) -> Result<Vec<u8>, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("empty audio payload".to_string());
    }
    let b64_part = if let Some(rest) = trimmed.strip_prefix("data:") {
        let Some((_, payload)) = rest.split_once(',') else {
            return Err("malformed audio data URL: missing comma separator".to_string());
        };
        payload
    } else {
        trimmed
    };
    let bytes = decode_base64(b64_part)?;
    if bytes.len() < 16 {
        return Err("decoded audio payload too short".to_string());
    }
    Ok(bytes)
}

/// Transcribes spoken audio bytes (`RIFF/WAVE` or CoreAudio-readable container) on the Apple Neural Engine.
pub fn transcribe_audio_bytes(audio_bytes: &[u8]) -> Result<VoiceTranscription, String> {
    if audio_bytes.len() < 16 {
        return Err("audio buffer too short".to_string());
    }
    let t0 = Instant::now();
    let is_riff_wav = audio_bytes.len() >= 44
        && &audio_bytes[0..4] == b"RIFF"
        && &audio_bytes[8..12] == b"WAVE";

    let (payload_to_write, ext, audio_ms) = if is_riff_wav {
        let prep = canonicalize_and_trim_wav(audio_bytes)?;
        if !prep.speech_detected {
            return Ok(VoiceTranscription {
                raw_text: String::new(),
                text: String::new(),
                audio_ms: prep.audio_ms,
                transcribe_ms: t0.elapsed().as_secs_f64() * 1000.0,
                engine: "macos-speech-analyzer-ane",
            });
        }
        (prep.wav_bytes, "wav", prep.audio_ms)
    } else {
        (audio_bytes.to_vec(), sniff_audio_ext(audio_bytes), 0)
    };

    #[cfg(target_os = "macos")]
    {
        let tmp_path = temp_audio_path("utt", ext);
        fs::write(&tmp_path, &payload_to_write)
            .map_err(|e| format!("failed to stage audio buffer: {e}"))?;

        let mgr = global_device_manager();
        let hints = mgr.speech_contextual_hints(40);
        let hint_field = hints
            .iter()
            .map(|h| h.replace(['|', '\t', '\n', '\r'], " "))
            .collect::<Vec<_>>()
            .join("|");
        let cmd = format!("{}\t{}", tmp_path.display(), hint_field);
        let worker_res = send_worker_command(&cmd);
        let _ = fs::remove_file(&tmp_path);

        let raw_text = worker_res?;
        SPEECH_WARMED.store(true, Ordering::Relaxed);
        let camel_idents = mgr.camel_case_identifiers();
        let normalized = normalize_voice_transcript(&raw_text, &camel_idents);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(VoiceTranscription {
            raw_text,
            text: normalized,
            audio_ms,
            transcribe_ms: elapsed_ms,
            engine: "macos-speech-analyzer-ane",
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (payload_to_write, ext, audio_ms);
        Err("local Neural Engine speech transcription requires macOS".to_string())
    }
}

fn sniff_audio_ext(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        "m4a"
    } else if bytes.starts_with(b"ID3") || (bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0) {
        "mp3"
    } else if bytes.starts_with(b"caff") {
        "caf"
    } else if bytes.starts_with(b"FORM") {
        "aiff"
    } else if bytes.starts_with(b"OggS") {
        "ogg"
    } else {
        "wav"
    }
}

/// Parses PCM16/24/32 or Float32 WAV, downmixes to mono, normalizes quiet mic gain, and trims outer silence.
pub fn canonicalize_and_trim_wav(raw: &[u8]) -> Result<PreparedWav, String> {
    if raw.len() < 44 || &raw[0..4] != b"RIFF" || &raw[8..12] != b"WAVE" {
        return Err("invalid RIFF/WAVE header".to_string());
    }
    let mut pos = 12usize;
    let mut audio_format = 0u16;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut data_slice: Option<&[u8]> = None;

    while pos + 8 <= raw.len() {
        let tag = &raw[pos..pos + 4];
        let chunk_len = u32::from_le_bytes(raw[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(chunk_len).min(raw.len());
        if tag == b"fmt " && body_end - body_start >= 16 {
            let fmt = &raw[body_start..body_end];
            let raw_fmt = u16::from_le_bytes(fmt[0..2].try_into().unwrap());
            channels = u16::from_le_bytes(fmt[2..4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(fmt[14..16].try_into().unwrap());
            audio_format = if raw_fmt == 0xFFFE && fmt.len() >= 26 {
                u16::from_le_bytes(fmt[24..26].try_into().unwrap())
            } else {
                raw_fmt
            };
        } else if tag == b"data" {
            data_slice = Some(&raw[body_start..body_end]);
            break;
        }
        let padded = chunk_len + (chunk_len & 1);
        pos = body_start.saturating_add(padded);
    }

    let data = data_slice.ok_or_else(|| "WAV missing data chunk".to_string())?;
    if channels == 0 || channels > 8 {
        return Err(format!("unsupported WAV channel count {channels}"));
    }
    if !(8_000..=192_000).contains(&sample_rate) {
        return Err(format!("unsupported WAV sample rate {sample_rate}"));
    }

    let ch = channels as usize;
    let bytes_per_sample = (bits_per_sample as usize) / 8;
    if bytes_per_sample == 0 {
        return Err("invalid WAV bits_per_sample".to_string());
    }
    let frame_bytes = bytes_per_sample * ch;
    let frame_count = data.len() / frame_bytes;
    let mut mono = Vec::with_capacity(frame_count);

    match (audio_format, bits_per_sample) {
        (1, 16) => {
            for f in 0..frame_count {
                let base = f * frame_bytes;
                let mut acc = 0i32;
                for c in 0..ch {
                    let off = base + c * 2;
                    acc += i16::from_le_bytes([data[off], data[off + 1]]) as i32;
                }
                mono.push((acc / ch as i32) as i16);
            }
        }
        (1, 24) => {
            for f in 0..frame_count {
                let base = f * frame_bytes;
                let mut acc = 0i32;
                for c in 0..ch {
                    let off = base + c * 3;
                    let v = ((data[off + 2] as i32) << 24
                        | (data[off + 1] as i32) << 16
                        | (data[off] as i32) << 8)
                        >> 16;
                    acc += v;
                }
                mono.push((acc / ch as i32).clamp(-32768, 32767) as i16);
            }
        }
        (1, 32) => {
            for f in 0..frame_count {
                let base = f * frame_bytes;
                let mut acc = 0i64;
                for c in 0..ch {
                    let off = base + c * 4;
                    let v = i32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                    acc += (v >> 16) as i64;
                }
                mono.push((acc / ch as i64).clamp(-32768, 32767) as i16);
            }
        }
        (3, 32) => {
            for f in 0..frame_count {
                let base = f * frame_bytes;
                let mut acc = 0.0f32;
                for c in 0..ch {
                    let off = base + c * 4;
                    let v = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                    if v.is_finite() {
                        acc += v;
                    }
                }
                let avg = (acc / ch as f32).clamp(-1.0, 1.0);
                mono.push((avg * 32767.0) as i16);
            }
        }
        _ => {
            return Err(format!(
                "unsupported WAV encoding format={audio_format} bits={bits_per_sample}"
            ));
        }
    }

    let raw_audio_ms = (mono.len() as u64 * 1000) / sample_rate as u64;
    if mono.is_empty() {
        return Ok(PreparedWav {
            wav_bytes: encode_pcm16_mono_wav(&[], sample_rate),
            audio_ms: 0,
            speech_detected: false,
        });
    }

    let peak_abs = mono
        .iter()
        .map(|&s| (s as i32).unsigned_abs())
        .max()
        .unwrap_or(0);
    if peak_abs < 55 {
        return Ok(PreparedWav {
            wav_bytes: encode_pcm16_mono_wav(&mono, sample_rate),
            audio_ms: raw_audio_ms,
            speech_detected: false,
        });
    }

    if peak_abs < 12_000 {
        let gain = (22_000.0 / peak_abs as f32).min(6.0);
        for s in &mut mono {
            *s = ((*s as f32 * gain).round() as i32).clamp(-32767, 32767) as i16;
        }
    }

    let norm_peak = mono
        .iter()
        .map(|&s| (s as i32).unsigned_abs())
        .max()
        .unwrap_or(1) as f32;
    let rms_gate = (norm_peak * 0.032).clamp(130.0, 620.0);
    let win = ((sample_rate as usize) / 50).max(1);
    let num_wins = mono.len().div_ceil(win);
    let mut first_voiced = None;
    let mut last_voiced = None;

    for w in 0..num_wins {
        let s0 = w * win;
        let s1 = (s0 + win).min(mono.len());
        let mut sum_sq = 0.0f64;
        for &sample in &mono[s0..s1] {
            let vf = sample as f64;
            sum_sq += vf * vf;
        }
        let rms = (sum_sq / (s1 - s0) as f64).sqrt() as f32;
        if rms >= rms_gate {
            if first_voiced.is_none() {
                first_voiced = Some(s0);
            }
            last_voiced = Some(s1);
        }
    }

    let Some(v_start) = first_voiced else {
        return Ok(PreparedWav {
            wav_bytes: encode_pcm16_mono_wav(&mono, sample_rate),
            audio_ms: raw_audio_ms,
            speech_detected: false,
        });
    };
    let v_end = last_voiced.unwrap_or(mono.len());
    let pad = (sample_rate as usize * 160) / 1000;
    let trim_start = v_start.saturating_sub(pad);
    let trim_end = (v_end + pad).min(mono.len());
    let trimmed = &mono[trim_start..trim_end];
    let trimmed_ms = (trimmed.len() as u64 * 1000) / sample_rate as u64;

    Ok(PreparedWav {
        wav_bytes: encode_pcm16_mono_wav(trimmed, sample_rate),
        audio_ms: trimmed_ms,
        speech_detected: true,
    })
}

pub fn encode_pcm16_mono_wav(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let data_bytes = (samples.len() * 2) as u32;
    let riff_size = 36u32.saturating_add(data_bytes);
    let byte_rate = sample_rate * 2;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[derive(Clone, Debug)]
struct PunctWord {
    lead: String,
    core: String,
    trail: String,
}

impl PunctWord {
    fn parse(tok: &str) -> Self {
        let bytes = tok.as_bytes();
        let mut s = 0usize;
        while s < bytes.len() && !bytes[s].is_ascii_alphanumeric() {
            s += 1;
        }
        let mut e = bytes.len();
        while e > s && !bytes[e - 1].is_ascii_alphanumeric() {
            e -= 1;
        }
        Self {
            lead: tok[..s].to_string(),
            core: tok[s..e].to_string(),
            trail: tok[e..].to_string(),
        }
    }

    fn render(&self) -> String {
        format!("{}{}{}", self.lead, self.core, self.trail)
    }
}

fn is_standalone_acronym_after_article(joined_low: &str) -> bool {
    matches!(
        joined_low,
        "nd" | "ui"
            | "ux"
            | "cv"
            | "id"
            | "os"
            | "ml"
            | "db"
            | "pr"
            | "ci"
            | "cd"
            | "qa"
            | "emr"
            | "ehr"
            | "crm"
            | "erp"
            | "sdk"
            | "ide"
            | "cli"
            | "gpu"
            | "cpu"
            | "ram"
            | "pdf"
            | "url"
            | "ssh"
            | "sql"
            | "css"
    )
}

/// Normalizes raw ASR transcripts: strips optional wake prefixes, collapses spelled acronyms,
/// repairs credential homophones ("undee certificate" -> "ND certificate"), and aligns camelCase
/// folder/project identifiers from the local DeviceIndex ("org prod" -> "orgProj").
pub fn normalize_voice_transcript(raw: &str, camel_identifiers: &[String]) -> String {
    let mut trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Strip explicit wake phrases ("Hey Kronos, ...").
    let lower_init = trimmed.to_ascii_lowercase();
    for wake in ["hey kronos", "hi kronos", "okay kronos", "ok kronos"] {
        if lower_init.starts_with(wake) {
            let after = &trimmed[wake.len()..];
            if after.is_empty()
                || after.starts_with(|c: char| c.is_ascii_whitespace() || matches!(c, ',' | '.' | ':'))
            {
                trimmed = after
                    .trim_start_matches(|c: char| c.is_ascii_whitespace() || matches!(c, ',' | '.' | ':'));
                break;
            }
        }
    }
    if trimmed.to_ascii_lowercase().starts_with("kronos") {
        let after = &trimmed["kronos".len()..];
        if after.starts_with(',')
            || after
                .split_whitespace()
                .next()
                .is_some_and(|w| {
                    matches!(
                        w.to_ascii_lowercase().as_str(),
                        "open"
                            | "launch"
                            | "create"
                            | "make"
                            | "build"
                            | "find"
                            | "where"
                            | "where's"
                            | "what"
                            | "switch"
                            | "show"
                            | "reveal"
                            | "please"
                    )
                })
        {
            trimmed = after
                .trim_start_matches(|c: char| c.is_ascii_whitespace() || matches!(c, ',' | '.' | ':'));
        }
    }

    let parsed: Vec<PunctWord> = trimmed.split_whitespace().map(PunctWord::parse).collect();
    if parsed.is_empty() {
        return String::new();
    }

    // Pass 1: Collapse adjacent spoken single-letter acronyms ("N D" -> "ND", "e m r" -> "EMR").
    let mut words: Vec<PunctWord> = Vec::with_capacity(parsed.len());
    let mut i = 0usize;
    while i < parsed.len() {
        let can_start = parsed[i].core.len() == 1
            && parsed[i].core.chars().all(|c| c.is_ascii_alphabetic())
            && (parsed[i].trail.is_empty()
                || (parsed[i].trail == "."
                    && parsed[i].core.chars().all(|c| c.is_ascii_uppercase())
                    && i + 1 < parsed.len()
                    && parsed[i + 1].core.len() == 1));
        if can_start {
            let start = i;
            i += 1;
            while i < parsed.len()
                && parsed[i].lead.is_empty()
                && parsed[i].core.len() == 1
                && parsed[i].core.chars().all(|c| c.is_ascii_alphabetic())
            {
                let prev_trail = &parsed[i - 1].trail;
                if !prev_trail.is_empty() && prev_trail != "." {
                    break;
                }
                i += 1;
            }
            let run = &parsed[start..i];
            if run.len() >= 2 {
                let first_low = run[0].core.to_ascii_lowercase();
                if first_low == "a" && run.len() >= 3 {
                    let tail_low: String =
                        run[1..].iter().map(|w| w.core.to_ascii_lowercase()).collect();
                    if is_standalone_acronym_after_article(&tail_low) {
                        words.push(run[0].clone());
                        let last_trail = clean_collapsed_trail(&run[run.len() - 1].trail, i == parsed.len());
                        words.push(PunctWord {
                            lead: String::new(),
                            core: tail_low.to_ascii_uppercase(),
                            trail: last_trail,
                        });
                        continue;
                    }
                }
                let joined_up: String = run.iter().map(|w| w.core.to_ascii_uppercase()).collect();
                let last_trail = clean_collapsed_trail(&run[run.len() - 1].trail, i == parsed.len());
                words.push(PunctWord {
                    lead: run[0].lead.clone(),
                    core: joined_up,
                    trail: last_trail,
                });
                continue;
            }
            words.push(run[0].clone());
        } else {
            words.push(parsed[i].clone());
            i += 1;
        }
    }

    // Pass 2: Repair spoken credential homophones ("undee certificate" -> "ND certificate").
    let has_credential_noun = words.iter().any(|w| {
        let low = w.core.to_ascii_lowercase();
        low.starts_with("cert")
            || low.starts_with("result")
            || low.starts_with("statement")
            || low.starts_with("diplom")
            || low.starts_with("transcript")
    });
    if has_credential_noun {
        for w in &mut words {
            if matches!(
                w.core.to_ascii_lowercase().as_str(),
                "undee" | "undy" | "endy" | "indie" | "indy"
            ) {
                w.core = "ND".to_string();
            }
        }
    }

    // Pre-split 2-part camelCase identifiers from DeviceIndex (e.g. "orgProj" -> ("org", "proj")).
    let camel_pairs: Vec<(&str, String, String)> = camel_identifiers
        .iter()
        .filter_map(|id| {
            let parts = split_camel_case_parts(id);
            if parts.len() == 2 && parts[0].len() >= 3 && parts[1].len() >= 3 {
                Some((id.as_str(), parts[0].clone(), parts[1].clone()))
            } else {
                None
            }
        })
        .collect();

    // Pass 3: Merge multi-word app names ("whats app" -> "WhatsApp") and 2-word camelCase folder/project names.
    let mut merged: Vec<PunctWord> = Vec::with_capacity(words.len());
    let mut k = 0usize;
    while k < words.len() {
        if k + 1 < words.len() && words[k].trail.is_empty() && words[k + 1].lead.is_empty() {
            let w0 = words[k].core.to_ascii_lowercase();
            let w1 = words[k + 1].core.to_ascii_lowercase();

            if (w0 == "whats" || w0 == "what's") && w1 == "app" {
                merged.push(PunctWord {
                    lead: words[k].lead.clone(),
                    core: "WhatsApp".to_string(),
                    trail: words[k + 1].trail.clone(),
                });
                k += 2;
                continue;
            }

            let exact_hit = camel_pairs
                .iter()
                .find(|(_, p0, p1)| w0 == *p0 && w1 == *p1)
                .map(|(ident, _, _)| *ident);
            let near_hit = exact_hit.or_else(|| {
                camel_pairs
                    .iter()
                    .find(|(_, p0, p1)| {
                        (w0 == *p0 && shares_prefix2_and_dist1(&w1, p1))
                            || (w1 == *p1 && shares_prefix2_and_dist1(&w0, p0))
                    })
                    .map(|(ident, _, _)| *ident)
            });

            if let Some(ident) = near_hit {
                merged.push(PunctWord {
                    lead: words[k].lead.clone(),
                    core: ident.to_string(),
                    trail: words[k + 1].trail.clone(),
                });
                k += 2;
                continue;
            }
        }
        merged.push(words[k].clone());
        k += 1;
    }

    merged
        .into_iter()
        .map(|w| w.render())
        .collect::<Vec<_>>()
        .join(" ")
}

fn clean_collapsed_trail(trail: &str, is_end_of_utterance: bool) -> String {
    if trail == "." && !is_end_of_utterance {
        String::new()
    } else {
        trail.to_string()
    }
}

fn shares_prefix2_and_dist1(a: &str, b: &str) -> bool {
    if a.len() < 3 || b.len() < 3 {
        return false;
    }
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    if ab[0] != bb[0] || ab[1] != bb[1] {
        return false;
    }
    levenshtein_ascii(a, b) <= 1
}

fn levenshtein_ascii(a: &str, b: &str) -> usize {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    if ab.len().abs_diff(bb.len()) > 2 {
        return 3;
    }
    let mut prev: Vec<usize> = (0..=bb.len()).collect();
    let mut curr = vec![0usize; bb.len() + 1];
    for (i, &ca) in ab.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in bb.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[bb.len()]
}

#[cfg(target_os = "macos")]
fn temp_audio_path(tag: &str, ext: &str) -> PathBuf {
    let seq = SPEECH_REQ_SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    std::env::temp_dir().join(format!(
        "kronos_speech_{tag}_{}_{seq}_{nanos}.{ext}",
        std::process::id()
    ))
}

#[cfg(target_os = "macos")]
fn fnv1a_u64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(target_os = "macos")]
fn ensure_compiled_speech_worker_bin() -> Result<PathBuf, String> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let bin_dir = home.join(".kronos").join("bin");
    fs::create_dir_all(&bin_dir)
        .map_err(|e| format!("failed to create {}: {e}", bin_dir.display()))?;

    let hash = fnv1a_u64(SWIFT_SPEECH_WORKER_SRC.as_bytes());
    let bin_path = bin_dir.join(format!("kronos_speech_ane_{hash:016x}"));
    if bin_path.exists() {
        return Ok(bin_path);
    }

    let pid = std::process::id();
    let src_path = bin_dir.join(format!("kronos_speech_ane_{hash:016x}_{pid}.swift"));
    let tmp_bin = bin_dir.join(format!("kronos_speech_ane_{hash:016x}_{pid}.tmp"));
    fs::write(&src_path, SWIFT_SPEECH_WORKER_SRC)
        .map_err(|e| format!("failed to write speech worker source: {e}"))?;

    let out = Command::new("/usr/bin/swiftc")
        .args(["-O", "-parse-as-library"])
        .arg(&src_path)
        .arg("-o")
        .arg(&tmp_bin)
        .output()
        .map_err(|e| format!("failed to invoke /usr/bin/swiftc: {e}"))?;
    let _ = fs::remove_file(&src_path);

    if !out.status.success() {
        let _ = fs::remove_file(&tmp_bin);
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(format!("swiftc failed compiling speech worker: {stderr}"));
    }

    fs::rename(&tmp_bin, &bin_path).map_err(|e| {
        let _ = fs::remove_file(&tmp_bin);
        format!("failed to install compiled speech worker: {e}")
    })?;
    Ok(bin_path)
}

#[cfg(target_os = "macos")]
fn spawn_speech_worker() -> Result<SpeechWorkerProcess, String> {
    let bin = ensure_compiled_speech_worker_bin()?;
    let mut child = Command::new(&bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to spawn {}: {e}", bin.display()))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "missing speech worker stdin".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "missing speech worker stdout".to_string())?;
    Ok(SpeechWorkerProcess {
        child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

#[cfg(target_os = "macos")]
fn send_worker_command(cmd_line: &str) -> Result<String, String> {
    let mut guard = worker_slot().lock().unwrap_or_else(|p| p.into_inner());
    for attempt in 0..2 {
        if guard.is_none() {
            *guard = Some(spawn_speech_worker()?);
        }
        let io_res = {
            let worker = guard.as_mut().unwrap();
            let write_ok = writeln!(worker.stdin, "{cmd_line}")
                .and_then(|()| worker.stdin.flush());
            if let Err(e) = write_ok {
                Err(format!("speech worker write error: {e}"))
            } else {
                let mut resp = String::new();
                match worker.stdout.read_line(&mut resp) {
                    Ok(0) => Err("speech worker closed stdout".to_string()),
                    Ok(_) => Ok(resp),
                    Err(e) => Err(format!("speech worker read error: {e}")),
                }
            }
        };

        match io_res {
            Ok(resp) => {
                let line = resp.trim_end_matches(['\r', '\n']);
                if let Some(rest) = line.strip_prefix("OK\t") {
                    let transcript = rest
                        .split_once('\t')
                        .map_or(rest, |(_, txt)| txt)
                        .trim()
                        .to_string();
                    return Ok(transcript);
                }
                if let Some(err) = line.strip_prefix("ERR\t") {
                    return Err(err.trim().to_string());
                }
                return Err(format!("unexpected speech worker reply: {line}"));
            }
            Err(e) => {
                *guard = None;
                if attempt == 1 {
                    return Err(e);
                }
            }
        }
    }
    Err("speech worker unreachable".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_canonicalize_trim_and_silence_gate() {
        let sr = 16_000u32;
        let silent = encode_pcm16_mono_wav(&vec![0i16; 8_000], sr);
        let prep_silent = canonicalize_and_trim_wav(&silent).expect("parse silent wav");
        assert!(!prep_silent.speech_detected);

        // 0.5s leading silence + 0.4s quiet 400Hz tone + 0.5s trailing silence
        let mut samples = vec![0i16; 8_000];
        for n in 0..6_400 {
            let phase = (n as f32 * 2.0 * std::f32::consts::PI * 400.0) / sr as f32;
            samples.push((phase.sin() * 3_200.0) as i16);
        }
        samples.extend(std::iter::repeat_n(0i16, 8_000));
        let raw_wav = encode_pcm16_mono_wav(&samples, sr);

        let prep = canonicalize_and_trim_wav(&raw_wav).expect("parse voiced wav");
        assert!(prep.speech_detected);
        // Trimmed duration should be ~400ms + 2*160ms pad = ~720ms (much shorter than 1400ms raw).
        assert!(prep.audio_ms >= 380 && prep.audio_ms <= 820);
        assert_eq!(&prep.wav_bytes[0..4], b"RIFF");
        assert_eq!(&prep.wav_bytes[8..12], b"WAVE");
    }

    #[test]
    fn test_voice_transcript_normalization_across_target_commands() {
        let camels = vec!["orgProj".to_string(), "kronosServer".to_string()];

        assert_eq!(
            normalize_voice_transcript(
                "Create a new folder in the documents org prod folder, then make a website about an e m r software.",
                &camels
            ),
            "Create a new folder in the documents orgProj folder, then make a website about an EMR software."
        );

        assert_eq!(
            normalize_voice_transcript("Open my undee certificate.", &camels),
            "Open my ND certificate."
        );

        assert_eq!(
            normalize_voice_transcript("Hey Kronos, open my N. D. certificate", &camels),
            "open my ND certificate"
        );

        assert_eq!(
            normalize_voice_transcript("Okay Kronos, open whats app", &camels),
            "open WhatsApp"
        );

        assert_eq!(
            normalize_voice_transcript("build a u i and call the a p i", &camels),
            "build a UI and call the API"
        );
    }
}
