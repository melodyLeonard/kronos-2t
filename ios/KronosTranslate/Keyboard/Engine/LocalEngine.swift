import Foundation

/// The bundled translation model, run in-process by the Kronos runtime (`libkronos_2t.a`).
///
/// A keyboard extension gets a small memory cap, so the weights are mapped read-only straight from
/// the `.k2tb` in the bundle (clean file-backed pages, which iOS does not count) and only the
/// activations, KV cache and tokenizer live on the heap.
final class LocalEngine {
    static let shared = LocalEngine()

    enum State { case idle, loading, ready, unavailable(String) }

    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// One translation in flight; cancelling stops decoding at the next token.
    final class Job {
        fileprivate var cancelled = false
        fileprivate var onPartial: ((String) -> Void)?
        fileprivate var latest = ""
        func cancel() { cancelled = true }
    }

    static let modelName = "Hy-MT2-1.8B"
    static let mode = "nf4g128"
    static var modelDir: URL? { Bundle.main.url(forResource: modelName, withExtension: nil) }
    static var isBundled: Bool { modelDir != nil }

    private(set) var state: State = .idle
    private(set) var loadMillis: Double = 0
    private var session: OpaquePointer?
    private let queue = DispatchQueue(label: "kronos.engine", qos: .userInitiated)
    private var waiters: [(State) -> Void] = []
    private var current: Job?
    private var started = 0, finished = 0, lastTokens = 0, lastMillis = 0.0
    /// Jobs started/finished and the last job's tokens and wall time, for diagnostics.
    var jobSummary: String { String(format: "jobs %d/%d · last %d tok %.1f s", started, finished, lastTokens, lastMillis) }

    /// Loads the session once, in the background. `completion` runs on the main thread.
    func load(completion: ((State) -> Void)? = nil) {
        if let completion { waiters.append(completion) }
        guard case .idle = state else { return }
        guard let dir = Self.modelDir else { finish(.unavailable("Model not bundled")); return }
        state = .loading
        queue.async { [self] in
            let t0 = Date()
            kronos_set_zero_copy_weights(1)
            // 512-token window bounds the KV cache (~16 KB/token); threads 0 = P-cores, more starves the UI.
            let sess = dir.path.withCString { kronos_session_create($0, Self.mode, 512, 0, 0) }
            let ms = Date().timeIntervalSince(t0) * 1000
            DispatchQueue.main.async {
                self.loadMillis = ms
                if let sess {
                    self.session = sess
                    self.finish(.ready)
                } else {
                    self.finish(.unavailable("Model failed to load"))
                }
            }
        }
    }

    private func finish(_ s: State) {
        state = s
        let w = waiters; waiters.removeAll()
        w.forEach { $0(s) }
    }

    /// Resolves once the model is ready or known to be unavailable.
    func ready() async -> State {
        if case .ready = state { return state }
        if case .unavailable = state { return state }
        return await withCheckedContinuation { c in load { c.resume(returning: $0) } }
    }

    /// Streams the cleaned translation-so-far to `onPartial` (engine queue) and returns the final text.
    /// Starting a new job cancels the previous one.
    func translate(_ text: String, from: String, to: String, context: [String], onPartial: @escaping (String) -> Void) async throws -> String {
        guard case .ready = await ready(), let session else { throw Failure(message: "Model unavailable") }
        let job = Job()
        job.onPartial = onPartial
        current?.cancel()
        current = job
        let ctx = (try? JSONSerialization.data(withJSONObject: context)).flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<String, Error>) in
                queue.async {
                    if job.cancelled { c.resume(throwing: CancellationError()); return }
                    self.started += 1
                    let t0 = Date()
                    let user = Unmanaged.passUnretained(job).toOpaque()
                    let n = text.withCString { t in to.withCString { tl in from.withCString { fl in ctx.withCString { cx in
                        kronos_session_translate(session, t, tl, fl, cx, { ptr, len, user in
                            guard let ptr, let user else { return true }
                            let job = Unmanaged<Job>.fromOpaque(user).takeUnretainedValue()
                            if job.cancelled { return false }
                            let s = String(decoding: UnsafeRawBufferPointer(start: ptr, count: len), as: UTF8.self)
                            job.latest = s
                            job.onPartial?(s)
                            return true
                        }, user)
                    } } } }
                    self.finished += 1; self.lastTokens = Int(n); self.lastMillis = Date().timeIntervalSince(t0)
                    if job.cancelled { c.resume(throwing: CancellationError()) }
                    else if n < 0 { c.resume(throwing: Failure(message: String(cString: kronos_session_last_error(session)))) }
                    else if job.latest.isEmpty { c.resume(throwing: Failure(message: "Empty translation")) }
                    else { c.resume(returning: job.latest) }
                }
            }
        } onCancel: {
            job.cancel()
        }
    }

    /// Memory headroom left under the extension's cap, and the process footprint, for diagnostics.
    static var memorySummary: String {
        let avail = Double(os_proc_available_memory()) / 1_048_576
        let rss = Double(kronos_process_footprint_bytes()) / 1_048_576
        return String(format: "free %.0f MB · used %.0f MB", avail, rss)
    }
}
