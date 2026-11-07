import Foundation

/// Client for Kronos `POST /translate`.
enum TranslateClient {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    static func translate(_ text: String, from: String, to code: String, tone: String, context: [String]) async throws -> String {
        guard let url = URL(string: Prefs.serverURL.trimmingCharacters(in: .whitespaces) + "/translate") else {
            throw Failure(message: "Bad server address")
        }
        var req = URLRequest(url: url, timeoutInterval: 30)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "text": text, "to": Languages.by(code: code).name, "from": Languages.by(code: from).name,
            "tone": tone, "context": context,
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            throw Failure(message: "Server error \((resp as? HTTPURLResponse)?.statusCode ?? 0)")
        }
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let out = obj["text"] as? String, !out.isEmpty else {
            throw Failure(message: "Empty translation")
        }
        return out
    }

    /// Round-trip check used by the app's Test button.
    static func ping() async -> Result<TimeInterval, Error> {
        let t0 = Date()
        do {
            _ = try await translate("hello", from: "EN", to: "FR", tone: "neutral", context: [])
            return .success(Date().timeIntervalSince(t0))
        } catch {
            return .failure(error)
        }
    }
}
