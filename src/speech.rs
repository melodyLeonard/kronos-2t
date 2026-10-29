//! Local on-device speech-to-text pipeline backed by macOS SpeechAnalyzer on the Apple Neural Engine.

#[cfg(target_os = "macos")]
use std::fs;
#[cfg(target_os = "macos")]
use std::io::{BufRead, BufReader, Write};
#[cfg(target_os = "macos")]
use std::os::unix::net::UnixStream;
#[cfg(target_os = "macos")]
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::process::Command;
#[cfg(target_os = "macos")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(target_os = "macos")]
use std::sync::{Mutex, OnceLock};
#[cfg(target_os = "macos")]
use std::thread;
use std::time::Instant;
#[cfg(target_os = "macos")]
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(target_os = "macos")]
use crate::device::global_device_manager;
use crate::device::split_camel_case_parts;
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

#[cfg(target_os = "macos")]
static SPEECH_WARMED: AtomicBool = AtomicBool::new(false);
#[cfg(target_os = "macos")]
static SPEECH_REQ_SEQ: AtomicU64 = AtomicU64::new(1);

// Packaged as a LaunchServices `.app` bundle (`KronosVoice.app`) so macOS TCC attributes microphone access
// directly to Kronos rather than inheriting a parent terminal or IDE host process's TCC state, and so
// WKWebView (which does not expose `navigator.mediaDevices` on macOS) and browser tabs share identical
// low-latency native hardware microphone capture and Neural Engine SpeechAnalyzer transcription.
#[cfg(target_os = "macos")]
const SWIFT_SPEECH_WORKER_SRC: &str = r#"import Foundation
import AppKit
import Speech
import AVFoundation
import Darwin

final class RecorderController: @unchecked Sendable {
    private let lock = NSLock()
    private var recorder: AVAudioRecorder? = nil
    private var activePath: String? = nil
    private var autoStoppedPath: String? = nil
    private var autoStoppedElapsedMs: Int = 0
    private var startedAt: TimeInterval = 0
    private var lastVoiceAt: TimeInterval = 0
    private var speechAccumMs: Double = 0
    private var heardSpeech: Bool = false
    private var currentLevel: Double = 0
    private var sessionSeq: UInt64 = 0
    private var meterTimer: DispatchSourceTimer? = nil

    func startRecording() -> String? {
        lock.lock()
        defer { lock.unlock() }

        meterTimer?.cancel()
        meterTimer = nil
        if let r = recorder {
            r.stop()
            recorder = nil
        }
        if let old = activePath {
            try? FileManager.default.removeItem(atPath: old)
            activePath = nil
        }
        if let oldAuto = autoStoppedPath {
            try? FileManager.default.removeItem(atPath: oldAuto)
            autoStoppedPath = nil
        }

        sessionSeq &+= 1
        let mySeq = sessionSeq
        let wavPath = NSTemporaryDirectory() + "kronos_mic_\(Darwin.getpid())_\(mySeq).wav"
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]
        let rec: AVAudioRecorder
        do {
            rec = try AVAudioRecorder(url: URL(fileURLWithPath: wavPath), settings: settings)
        } catch {
            return "AVAudioRecorder init failed: \(error.localizedDescription)"
        }
        rec.isMeteringEnabled = true
        guard rec.prepareToRecord(), rec.record() else {
            try? FileManager.default.removeItem(atPath: wavPath)
            return "Unable to start macOS microphone capture"
        }

        let now = ProcessInfo.processInfo.systemUptime
        recorder = rec
        activePath = wavPath
        autoStoppedPath = nil
        autoStoppedElapsedMs = 0
        startedAt = now
        lastVoiceAt = now
        speechAccumMs = 0
        heardSpeech = false
        currentLevel = 0

        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .userInteractive))
        timer.schedule(deadline: .now() + .milliseconds(60), repeating: .milliseconds(60))
        timer.setEventHandler { [weak self] in
            guard let self = self else { return }
            self.lock.lock()
            defer { self.lock.unlock() }
            guard self.sessionSeq == mySeq, let r = self.recorder, r.isRecording else { return }
            r.updateMeters()
            let avgDb = Double(r.averagePower(forChannel: 0))
            let peakDb = Double(r.peakPower(forChannel: 0))
            let norm = min(1.0, max(0.0, (peakDb + 50.0) / 45.0))
            self.currentLevel = norm
            let t = ProcessInfo.processInfo.systemUptime
            if peakDb > -32.0 || avgDb > -37.0 {
                self.speechAccumMs += 60.0
                self.lastVoiceAt = t
                if self.speechAccumMs >= 180.0 {
                    self.heardSpeech = true
                }
            }
            let elapsed = t - self.startedAt
            if (self.heardSpeech && (t - self.lastVoiceAt) >= 1.35 && elapsed >= 0.90) || elapsed >= 45.0 {
                r.stop()
                self.recorder = nil
                self.meterTimer?.cancel()
                self.meterTimer = nil
                self.autoStoppedPath = self.activePath
                self.activePath = nil
                self.autoStoppedElapsedMs = Int(elapsed * 1000.0)
            }
        }
        timer.resume()
        meterTimer = timer
        return nil
    }

    func pollStatus() -> String {
        lock.lock()
        defer { lock.unlock() }
        if let readyPath = autoStoppedPath {
            let ms = autoStoppedElapsedMs
            autoStoppedPath = nil
            autoStoppedElapsedMs = 0
            return "WAV_READY\t\(readyPath)\t\(ms)"
        }
        if let r = recorder, r.isRecording {
            let elapsedMs = Int((ProcessInfo.processInfo.systemUptime - startedAt) * 1000.0)
            let hs = heardSpeech ? 1 : 0
            return String(format: "RECORDING\t%d\t%.3f\t%d", elapsedMs, currentLevel, hs)
        }
        return "IDLE"
    }

    func stopRecording(cancel: Bool) -> String {
        lock.lock()
        defer { lock.unlock() }
        sessionSeq &+= 1
        meterTimer?.cancel()
        meterTimer = nil
        if let r = recorder {
            let elapsedMs = Int((ProcessInfo.processInfo.systemUptime - startedAt) * 1000.0)
            r.stop()
            recorder = nil
            let path = activePath
            activePath = nil
            if cancel {
                if let p = path { try? FileManager.default.removeItem(atPath: p) }
                return "CANCELLED"
            } else if let p = path {
                return "WAV_READY\t\(p)\t\(elapsedMs)"
            }
        }
        if let readyPath = autoStoppedPath {
            let ms = autoStoppedElapsedMs
            autoStoppedPath = nil
            autoStoppedElapsedMs = 0
            if cancel {
                try? FileManager.default.removeItem(atPath: readyPath)
                return "CANCELLED"
            } else {
                return "WAV_READY\t\(readyPath)\t\(ms)"
            }
        }
        return cancel ? "CANCELLED" : "IDLE"
    }
}

@main
struct KronosSpeechWorker {
    static let recorderCtrl = RecorderController()

    static func sanitize(_ s: String) -> String {
        s.replacingOccurrences(of: "\r", with: " ")
         .replacingOccurrences(of: "\n", with: " ")
         .replacingOccurrences(of: "\t", with: " ")
         .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func bundleHashTag() -> String {
        if let url = Bundle.main.url(forResource: "src_hash", withExtension: "txt"),
           let txt = try? String(contentsOf: url, encoding: .utf8) {
            return sanitize(txt)
        }
        return "unknown"
    }

    static func defaultSocketPath() -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let dir = home + "/.kronos"
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let candidate = dir + "/voice.sock"
        if candidate.utf8.count < 96 {
            return candidate
        }
        return "/tmp/kronos_voice.sock"
    }

    static func bindUnixSocket(_ path: String) -> Int32 {
        unlink(path)
        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        if fd < 0 { return -1 }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let utf8 = Array(path.utf8CString)
        let maxLen = MemoryLayout.size(ofValue: addr.sun_path)
        if utf8.count > maxLen {
            Darwin.close(fd)
            return -1
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: maxLen) { dst in
                for i in 0..<utf8.count { dst[i] = utf8[i] }
            }
        }
        let bindRes = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.bind(fd, sa, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        if bindRes != 0 || Darwin.listen(fd, 24) != 0 {
            Darwin.close(fd)
            return -1
        }
        _ = Darwin.chmod(path, 0o600)
        return fd
    }

    static func readLineFromFd(_ fd: Int32) -> String? {
        var bytes: [UInt8] = []
        var chunk = [UInt8](repeating: 0, count: 1024)
        while bytes.count < 65536 {
            let n = Darwin.read(fd, &chunk, chunk.count)
            if n <= 0 { break }
            if let nlIdx = chunk[0..<n].firstIndex(of: 10) {
                bytes.append(contentsOf: chunk[0..<nlIdx])
                break
            }
            bytes.append(contentsOf: chunk[0..<n])
        }
        if bytes.isEmpty { return nil }
        return String(bytes: bytes, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func writeLineToFd(_ fd: Int32, _ line: String) {
        var one: Int32 = 1
        _ = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
        let payload = Array((line + "\n").utf8)
        var sent = 0
        while sent < payload.count {
            let n = payload.withUnsafeBytes { raw in
                Darwin.write(fd, raw.baseAddress!.advanced(by: sent), payload.count - sent)
            }
            if n <= 0 { break }
            sent += n
        }
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

    static func handleCommand(_ line: String, locale: Locale, sockPath: String) async -> (String, Bool) {
        if line == "QUIT" {
            _ = recorderCtrl.stopRecording(cancel: true)
            unlink(sockPath)
            return ("OK\tBYE", true)
        }
        if line == "PING" {
            let auth = AVCaptureDevice.authorizationStatus(for: .audio).rawValue
            return ("OK\tPONG\t\(bundleHashTag())\t\(auth)", false)
        }
        if line == "REC_STATUS" {
            return (recorderCtrl.pollStatus(), false)
        }
        if line.hasPrefix("REC_STOP") {
            let cancel = line.hasSuffix("\t1")
            return (recorderCtrl.stopRecording(cancel: cancel), false)
        }
        if line == "REC_START" {
            let status = AVCaptureDevice.authorizationStatus(for: .audio)
            if status == .notDetermined {
                let granted = await AVCaptureDevice.requestAccess(for: .audio)
                if !granted {
                    return ("DENIED\tMicrophone permission was not granted for Kronos", false)
                }
            } else if status != .authorized {
                return ("DENIED\tEnable Kronos under macOS System Settings -> Privacy & Security -> Microphone", false)
            }
            if let msg = recorderCtrl.startRecording() {
                return ("ERR\t" + sanitize(msg), false)
            }
            return ("OK\tRECORDING", false)
        }
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
            return (String(format: "OK\t%.2f\tREADY", ms), false)
        }

        let body = line.hasPrefix("TRANSCRIBE\t")
            ? String(line.dropFirst("TRANSCRIBE\t".count))
            : line
        let parts = body.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
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
            return (String(format: "OK\t%.2f\t%@", ms, sanitize(text)), false)
        } catch {
            return ("ERR\t" + sanitize(String(describing: error)), false)
        }
    }

    static func main() {
        Darwin.signal(SIGPIPE, SIG_IGN)
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)

        let sockPath = defaultSocketPath()
        let serverFd = bindUnixSocket(sockPath)
        if serverFd < 0 {
            exit(1)
        }

        DispatchQueue.global(qos: .userInitiated).async {
            let sem = DispatchSemaphore(value: 0)
            var resolvedLocale = Locale(identifier: "en_US")
            Task {
                resolvedLocale = await resolveLocale()
                sem.signal()
            }
            sem.wait()
            let locale = resolvedLocale

            while true {
                let clientFd = Darwin.accept(serverFd, nil, nil)
                if clientFd < 0 { continue }
                Task {
                    guard let reqLine = readLineFromFd(clientFd), !reqLine.isEmpty else {
                        Darwin.close(clientFd)
                        return
                    }
                    let (reply, shouldExit) = await handleCommand(reqLine, locale: locale, sockPath: sockPath)
                    writeLineToFd(clientFd, reply)
                    Darwin.close(clientFd)
                    if shouldExit {
                        Darwin.close(serverFd)
                        exit(0)
                    }
                }
            }
        }

        app.run()
    }
}
"#;

#[cfg(target_os = "macos")]
static DAEMON_BOOT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
#[cfg(target_os = "macos")]
static DAEMON_VERIFIED: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "macos")]
fn boot_lock() -> &'static Mutex<()> {
    DAEMON_BOOT_LOCK.get_or_init(|| Mutex::new(()))
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
        let cmd = format!("TRANSCRIBE\t{}\t{}", tmp_path.display(), hint_field);
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
fn voice_socket_path() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        let dir = home.join(".kronos");
        let _ = fs::create_dir_all(&dir);
        let cand = dir.join("voice.sock");
        if cand.to_string_lossy().len() < 96 {
            return cand;
        }
    }
    PathBuf::from("/tmp/kronos_voice.sock")
}

#[cfg(target_os = "macos")]
fn expected_worker_hash_hex() -> String {
    format!("{:016x}", fnv1a_u64(SWIFT_SPEECH_WORKER_SRC.as_bytes()))
}

#[cfg(target_os = "macos")]
fn ensure_compiled_voice_app_bundle() -> Result<PathBuf, String> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let bin_dir = home.join(".kronos").join("bin");
    fs::create_dir_all(&bin_dir)
        .map_err(|e| format!("failed to create {}: {e}", bin_dir.display()))?;

    let app_dir = bin_dir.join("KronosVoice.app");
    let contents_dir = app_dir.join("Contents");
    let macos_dir = contents_dir.join("MacOS");
    let res_dir = contents_dir.join("Resources");
    let exe_path = macos_dir.join("kronos_voice");
    let stamp_path = res_dir.join("src_hash.txt");
    let plist_path = contents_dir.join("Info.plist");

    let expected_hash = expected_worker_hash_hex();
    if exe_path.exists()
        && plist_path.exists()
        && fs::read_to_string(&stamp_path)
            .map(|s| s.trim() == expected_hash)
            .unwrap_or(false)
    {
        return Ok(app_dir);
    }

    fs::create_dir_all(&macos_dir)
        .map_err(|e| format!("failed to create {}: {e}", macos_dir.display()))?;
    fs::create_dir_all(&res_dir)
        .map_err(|e| format!("failed to create {}: {e}", res_dir.display()))?;

    let pid = std::process::id();
    let src_path = bin_dir.join(format!("kronos_voice_{expected_hash}_{pid}.swift"));
    let tmp_bin = bin_dir.join(format!("kronos_voice_{expected_hash}_{pid}.tmp"));
    fs::write(&src_path, SWIFT_SPEECH_WORKER_SRC)
        .map_err(|e| format!("failed to write voice worker source: {e}"))?;

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
        return Err(format!("swiftc failed compiling KronosVoice.app: {stderr}"));
    }

    fs::rename(&tmp_bin, &exe_path).map_err(|e| {
        let _ = fs::remove_file(&tmp_bin);
        format!("failed to install KronosVoice binary: {e}")
    })?;

    let info_plist = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.melodyleonard.kronos.voice</string>
    <key>CFBundleName</key>
    <string>Kronos</string>
    <key>CFBundleDisplayName</key>
    <string>Kronos</string>
    <key>CFBundleExecutable</key>
    <string>kronos_voice</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Kronos uses the microphone for local on-device voice commands.</string>
</dict>
</plist>
"#;
    fs::write(&plist_path, info_plist)
        .map_err(|e| format!("failed to write KronosVoice Info.plist: {e}"))?;
    fs::write(&stamp_path, format!("{expected_hash}\n"))
        .map_err(|e| format!("failed to write KronosVoice version stamp: {e}"))?;

    let sign_out = Command::new("/usr/bin/codesign")
        .args(["--force", "--deep", "--sign", "-"])
        .arg(&app_dir)
        .output()
        .map_err(|e| format!("failed to invoke /usr/bin/codesign: {e}"))?;
    if !sign_out.status.success() {
        let stderr = String::from_utf8_lossy(&sign_out.stderr).trim().to_string();
        return Err(format!("codesign failed for KronosVoice.app: {stderr}"));
    }
    Ok(app_dir)
}

#[cfg(target_os = "macos")]
fn exchange_socket_line(sock_path: &Path, cmd_line: &str, timeout: Duration) -> Result<String, String> {
    let mut stream = UnixStream::connect(sock_path)
        .map_err(|e| format!("connect {}: {e}", sock_path.display()))?;
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
    writeln!(stream, "{cmd_line}").map_err(|e| format!("socket write: {e}"))?;
    stream.flush().map_err(|e| format!("socket flush: {e}"))?;
    let mut reader = BufReader::new(stream);
    let mut resp = String::new();
    let n = reader
        .read_line(&mut resp)
        .map_err(|e| format!("socket read: {e}"))?;
    if n == 0 {
        return Err("voice daemon closed socket without reply".to_string());
    }
    Ok(resp.trim_end_matches(['\r', '\n']).to_string())
}

#[cfg(target_os = "macos")]
fn ensure_voice_daemon_running() -> Result<PathBuf, String> {
    let sock_path = voice_socket_path();
    if DAEMON_VERIFIED.load(Ordering::Relaxed) && UnixStream::connect(&sock_path).is_ok() {
        return Ok(sock_path);
    }

    let _guard = boot_lock().lock().unwrap_or_else(|p| p.into_inner());
    let expected_hash = expected_worker_hash_hex();
    if let Ok(pong) = exchange_socket_line(&sock_path, "PING", Duration::from_secs(2)) {
        let parts: Vec<&str> = pong.split('\t').collect();
        if parts.len() >= 3 && parts[0] == "OK" && parts[1] == "PONG" && parts[2] == expected_hash {
            DAEMON_VERIFIED.store(true, Ordering::Relaxed);
            return Ok(sock_path);
        }
        let _ = exchange_socket_line(&sock_path, "QUIT", Duration::from_secs(1));
        thread::sleep(Duration::from_millis(120));
    }

    let app_dir = ensure_compiled_voice_app_bundle()?;
    let _ = fs::remove_file(&sock_path);
    let status = Command::new("/usr/bin/open")
        .args(["-g", "-a"])
        .arg(&app_dir)
        .status()
        .map_err(|e| format!("failed to launch {}: {e}", app_dir.display()))?;
    if !status.success() {
        return Err(format!("LaunchServices failed starting {}", app_dir.display()));
    }

    let deadline = Instant::now() + Duration::from_secs(6);
    while Instant::now() < deadline {
        if let Ok(pong) = exchange_socket_line(&sock_path, "PING", Duration::from_secs(2)) {
            if pong.starts_with("OK\tPONG") {
                DAEMON_VERIFIED.store(true, Ordering::Relaxed);
                return Ok(sock_path);
            }
        }
        thread::sleep(Duration::from_millis(45));
    }
    Err("timed out waiting for KronosVoice.app socket readiness".to_string())
}

#[cfg(target_os = "macos")]
fn send_raw_daemon_command(cmd_line: &str, timeout: Duration) -> Result<String, String> {
    for attempt in 0..2 {
        let sock_path = ensure_voice_daemon_running()?;
        match exchange_socket_line(&sock_path, cmd_line, timeout) {
            Ok(resp) => return Ok(resp),
            Err(e) => {
                DAEMON_VERIFIED.store(false, Ordering::Relaxed);
                if attempt == 1 {
                    return Err(e);
                }
            }
        }
    }
    Err("KronosVoice daemon unreachable".to_string())
}

#[cfg(target_os = "macos")]
fn send_worker_command(cmd_line: &str) -> Result<String, String> {
    let line = send_raw_daemon_command(cmd_line, Duration::from_secs(30))?;
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
    Err(format!("unexpected voice daemon reply: {line}"))
}

#[cfg(target_os = "macos")]
fn transcribe_completed_mic_wav(wav_path_str: &str) -> Result<String, String> {
    let wav_path = Path::new(wav_path_str);
    let wav_bytes = fs::read(wav_path)
        .map_err(|e| format!("failed reading recorded audio {}: {e}", wav_path.display()))?;
    let _ = fs::remove_file(wav_path);
    let vt = transcribe_audio_bytes(&wav_bytes)?;
    Ok(format!(
        r#"{{"status":"done","text":{},"raw_text":{},"audio_ms":{},"transcribe_ms":{:.2},"engine":{}}}"#,
        quote(&vt.text),
        quote(&vt.raw_text),
        vt.audio_ms,
        vt.transcribe_ms,
        quote(vt.engine),
    ))
}

pub fn start_mic_recording_json() -> Result<String, String> {
    #[cfg(target_os = "macos")]
    {
        let reply = send_raw_daemon_command("REC_START", Duration::from_secs(60))?;
        if reply.starts_with("OK\tRECORDING") {
            return Ok(r#"{"status":"recording"}"#.to_string());
        }
        if let Some(msg) = reply.strip_prefix("DENIED\t") {
            return Ok(format!(
                r#"{{"status":"denied","message":{}}}"#,
                quote(msg.trim())
            ));
        }
        if let Some(err) = reply.strip_prefix("ERR\t") {
            return Err(err.trim().to_string());
        }
        Err(format!("unexpected REC_START reply: {reply}"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("native microphone capture requires macOS".to_string())
    }
}

pub fn poll_mic_recording_json() -> Result<String, String> {
    #[cfg(target_os = "macos")]
    {
        let reply = send_raw_daemon_command("REC_STATUS", Duration::from_secs(5))?;
        if let Some(rest) = reply.strip_prefix("RECORDING\t") {
            let parts: Vec<&str> = rest.split('\t').collect();
            let elapsed_ms: u64 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
            let level: f32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let heard = parts.get(2).copied() == Some("1");
            return Ok(format!(
                r#"{{"status":"recording","elapsed_ms":{elapsed_ms},"level":{level:.3},"heard_speech":{heard}}}"#
            ));
        }
        if let Some(rest) = reply.strip_prefix("WAV_READY\t") {
            let wav_path = rest.split('\t').next().unwrap_or("").trim();
            return transcribe_completed_mic_wav(wav_path);
        }
        Ok(r#"{"status":"idle"}"#.to_string())
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("native microphone capture requires macOS".to_string())
    }
}

pub fn stop_mic_recording_json(cancel: bool) -> Result<String, String> {
    #[cfg(target_os = "macos")]
    {
        let cmd = if cancel { "REC_STOP\t1" } else { "REC_STOP\t0" };
        let reply = send_raw_daemon_command(cmd, Duration::from_secs(10))?;
        if reply == "CANCELLED" {
            return Ok(r#"{"status":"cancelled"}"#.to_string());
        }
        if let Some(rest) = reply.strip_prefix("WAV_READY\t") {
            let wav_path = rest.split('\t').next().unwrap_or("").trim();
            return transcribe_completed_mic_wav(wav_path);
        }
        Ok(r#"{"status":"idle"}"#.to_string())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = cancel;
        Err("native microphone capture requires macOS".to_string())
    }
}

pub fn open_macos_microphone_settings() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        Command::new("/usr/bin/open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
            .spawn()
            .map(|_| ())
            .map_err(|e| format!("failed opening System Settings: {e}"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("System Settings requires macOS".to_string())
    }
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

    #[cfg(target_os = "macos")]
    #[test]
    fn test_kronos_voice_app_bundle_compiles_and_signs() {
        let app_dir = ensure_compiled_voice_app_bundle().expect("compile KronosVoice.app");
        assert!(app_dir.join("Contents/MacOS/kronos_voice").exists());
        assert!(app_dir.join("Contents/Info.plist").exists());
        assert!(app_dir.join("Contents/Resources/src_hash.txt").exists());
    }
}
