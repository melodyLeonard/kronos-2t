import Foundation

/// Settings shared between the app and the keyboard through the app group.
enum Prefs {
    static let group = "group.com.melodyleonard.kronostranslate"
    static let contextMax = 5
    static let defaults: UserDefaults = UserDefaults(suiteName: group) ?? .standard

    static var serverURL: String {
        get { defaults.string(forKey: "serverURL") ?? "http://192.168.1.70:18085" }
        set { defaults.set(newValue, forKey: "serverURL") }
    }
    static var targetCode: String {
        get { defaults.string(forKey: "targetCode") ?? "FR" }
        set { defaults.set(newValue, forKey: "targetCode") }
    }
    static var tone: String {
        get { defaults.string(forKey: "tone") ?? "casual" }
        set { defaults.set(newValue, forKey: "tone") }
    }
    static var contextOn: Bool {
        get { defaults.object(forKey: "contextOn") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "contextOn") }
    }
    static var recents: [String] {
        get { defaults.stringArray(forKey: "recents") ?? ["FR", "ES", "DE", "YO"] }
        set { defaults.set(Array(newValue.prefix(4)), forKey: "recents") }
    }
    /// Last sent messages, oldest first, used as translation context.
    static var context: [String] {
        get { defaults.stringArray(forKey: "context") ?? [] }
        set { defaults.set(Array(newValue.suffix(contextMax)), forKey: "context") }
    }

    static func touchRecent(_ code: String) {
        var r = recents.filter { $0 != code }
        r.insert(code, at: 0)
        recents = r
    }
    static func remember(sent text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        context = context + [t]
    }
}

struct Language: Identifiable, Hashable {
    let code: String
    let name: String
    let native: String
    var id: String { code }
}

enum Languages {
    static let all: [Language] = [
        .init(code: "FR", name: "French", native: "Français"),
        .init(code: "ES", name: "Spanish", native: "Español"),
        .init(code: "PT", name: "Portuguese", native: "Português"),
        .init(code: "DE", name: "German", native: "Deutsch"),
        .init(code: "IT", name: "Italian", native: "Italiano"),
        .init(code: "NL", name: "Dutch", native: "Nederlands"),
        .init(code: "AR", name: "Arabic", native: "العربية"),
        .init(code: "TR", name: "Turkish", native: "Türkçe"),
        .init(code: "RU", name: "Russian", native: "Русский"),
        .init(code: "ZH", name: "Chinese", native: "中文"),
        .init(code: "JA", name: "Japanese", native: "日本語"),
        .init(code: "KO", name: "Korean", native: "한국어"),
        .init(code: "HI", name: "Hindi", native: "हिन्दी"),
        .init(code: "YO", name: "Yoruba", native: "Yorùbá"),
        .init(code: "IG", name: "Igbo", native: "Igbo"),
        .init(code: "HA", name: "Hausa", native: "Hausa"),
        .init(code: "SW", name: "Swahili", native: "Kiswahili"),
        .init(code: "EN", name: "English", native: "English"),
    ]
    static func by(code: String) -> Language { all.first { $0.code == code } ?? all[0] }
}
