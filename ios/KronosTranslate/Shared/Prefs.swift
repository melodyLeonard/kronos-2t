import Foundation

/// Settings shared between the app and the keyboard through the app group.
enum Prefs {
    static let group = "group.com.melodyleonard.kronostranslate"
    static let contextMax = 5
    static let githubURL = "https://github.com/melodyLeonard/kronos-2t"
    static let defaults: UserDefaults = UserDefaults(suiteName: group) ?? .standard

    static var serverURL: String {
        get { defaults.string(forKey: "serverURL") ?? "http://192.168.1.70:18085" }
        set { defaults.set(newValue, forKey: "serverURL") }
    }
    /// Language the user types in; also picks the key layout.
    static var sourceCode: String {
        get { defaults.string(forKey: "sourceCode") ?? "EN" }
        set { defaults.set(newValue, forKey: "sourceCode") }
    }
    static var targetCode: String {
        get { defaults.string(forKey: "targetCode") ?? "FR" }
        set { defaults.set(newValue, forKey: "targetCode") }
    }
    static var contextOn: Bool {
        get { defaults.object(forKey: "contextOn") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "contextOn") }
    }
    static var recents: [String] {
        get { defaults.stringArray(forKey: "recents") ?? ["FR", "ES", "DE", "YO"] }
        set { defaults.set(Array(newValue.prefix(4)), forKey: "recents") }
    }
    static var recentEmoji: [String] {
        get { defaults.stringArray(forKey: "recentEmoji") ?? [] }
        set { defaults.set(Array(newValue.prefix(32)), forKey: "recentEmoji") }
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
    static func touchEmoji(_ e: String) {
        var r = recentEmoji.filter { $0 != e }
        r.insert(e, at: 0)
        recentEmoji = r
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
        .init(code: "EN", name: "English", native: "English"),
        .init(code: "FR", name: "French", native: "Français"),
        .init(code: "ES", name: "Spanish", native: "Español"),
        .init(code: "PT", name: "Portuguese", native: "Português"),
        .init(code: "DE", name: "German", native: "Deutsch"),
        .init(code: "IT", name: "Italian", native: "Italiano"),
        .init(code: "NL", name: "Dutch", native: "Nederlands"),
        .init(code: "PL", name: "Polish", native: "Polski"),
        .init(code: "CS", name: "Czech", native: "Čeština"),
        .init(code: "UK", name: "Ukrainian", native: "Українська"),
        .init(code: "RU", name: "Russian", native: "Русский"),
        .init(code: "TR", name: "Turkish", native: "Türkçe"),
        .init(code: "AR", name: "Arabic", native: "العربية"),
        .init(code: "FA", name: "Persian", native: "فارسی"),
        .init(code: "HE", name: "Hebrew", native: "עברית"),
        .init(code: "HI", name: "Hindi", native: "हिन्दी"),
        .init(code: "BN", name: "Bengali", native: "বাংলা"),
        .init(code: "UR", name: "Urdu", native: "اردو"),
        .init(code: "TA", name: "Tamil", native: "தமிழ்"),
        .init(code: "TE", name: "Telugu", native: "తెలుగు"),
        .init(code: "ZH", name: "Chinese", native: "中文"),
        .init(code: "JA", name: "Japanese", native: "日本語"),
        .init(code: "KO", name: "Korean", native: "한국어"),
        .init(code: "VI", name: "Vietnamese", native: "Tiếng Việt"),
        .init(code: "TH", name: "Thai", native: "ไทย"),
        .init(code: "ID", name: "Indonesian", native: "Bahasa Indonesia"),
        .init(code: "MS", name: "Malay", native: "Bahasa Melayu"),
        .init(code: "FIL", name: "Filipino", native: "Filipino"),
        .init(code: "YO", name: "Yoruba", native: "Yorùbá"),
        .init(code: "IG", name: "Igbo", native: "Igbo"),
        .init(code: "HA", name: "Hausa", native: "Hausa"),
        .init(code: "SW", name: "Swahili", native: "Kiswahili"),
    ]
    static func by(code: String) -> Language { all.first { $0.code == code } ?? all[0] }
}
