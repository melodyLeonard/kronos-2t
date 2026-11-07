import Foundation

/// Client for Kronos `POST /translate` (persistent keep-alive + SSE token streaming).
enum TranslateClient {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// Low-latency persistent session so consecutive keystrokes reuse the same warm TCP socket.
    private static let session: URLSession = {
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 15
        cfg.timeoutIntervalForResource = 30
        cfg.waitsForConnectivity = false
        cfg.networkServiceType = .responsiveData
        cfg.httpMaximumConnectionsPerHost = 2
        return URLSession(configuration: cfg)
    }()

    /// Non-streaming one-shot translation (used by `ping()` and fallback).
    static func translate(_ text: String, from: String, to code: String, context: [String]) async throws -> String {
        let req = try makeRequest(text: text, from: from, to: code, context: context, stream: false)
        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            throw Failure(message: "Server error \((resp as? HTTPURLResponse)?.statusCode ?? 0)")
        }
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let out = obj["text"] as? String, !out.isEmpty else {
            throw Failure(message: "Empty translation")
        }
        return out
    }

    /// Streams incremental translation tokens over SSE, invoking `onPartial` on the main actor as tokens arrive.
    static func translateStream(
        _ text: String,
        from: String,
        to code: String,
        context: [String],
        onPartial: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        let req = try makeRequest(text: text, from: from, to: code, context: context, stream: true)
        let (bytes, resp) = try await session.bytes(for: req)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            throw Failure(message: "Server error \((resp as? HTTPURLResponse)?.statusCode ?? 0)")
        }
        var latest = ""
        for try await line in bytes.lines {
            try Task.checkCancellation()
            guard line.hasPrefix("data:") else { continue }
            let raw = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            if raw == "[DONE]" { break }
            guard let data = raw.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            if let err = obj["error"] as? String {
                throw Failure(message: err)
            }
            if let t = obj["text"] as? String, !t.isEmpty {
                latest = t
                await onPartial(t)
            }
            if (obj["done"] as? Bool) == true { break }
        }
        if latest.isEmpty {
            throw Failure(message: "Empty translation")
        }
        return latest
    }

    private static func makeRequest(text: String, from: String, to code: String, context: [String], stream: Bool) throws -> URLRequest {
        guard let url = URL(string: Prefs.serverURL.trimmingCharacters(in: .whitespaces) + "/translate") else {
            throw Failure(message: "Bad server address")
        }
        var req = URLRequest(url: url, timeoutInterval: 15)
        req.httpMethod = "POST"
        req.networkServiceType = .responsiveData
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("keep-alive", forHTTPHeaderField: "Connection")
        let body: [String: Any] = [
            "text": text,
            "to": Languages.by(code: code).name,
            "from": Languages.by(code: from).name,
            "tone": "neutral",
            "context": context,
            "stream": stream,
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        return req
    }

    /// Round-trip check used by the app's Test button.
    static func ping() async -> Result<TimeInterval, Error> {
        let t0 = Date()
        do {
            _ = try await translate("hello", from: "EN", to: "FR", context: [])
            return .success(Date().timeIntervalSince(t0))
        } catch {
            return .failure(error)
        }
    }
}
