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

/// Every language the server can serve: Hy-MT2's 38 plus the ones the general model covers.
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
        .init(code: "SK", name: "Slovak", native: "Slovenčina"),
        .init(code: "HU", name: "Hungarian", native: "Magyar"),
        .init(code: "RO", name: "Romanian", native: "Română"),
        .init(code: "SV", name: "Swedish", native: "Svenska"),
        .init(code: "DA", name: "Danish", native: "Dansk"),
        .init(code: "NO", name: "Norwegian", native: "Norsk"),
        .init(code: "FI", name: "Finnish", native: "Suomi"),
        .init(code: "EL", name: "Greek", native: "Ελληνικά"),
        .init(code: "UK", name: "Ukrainian", native: "Українська"),
        .init(code: "RU", name: "Russian", native: "Русский"),
        .init(code: "KK", name: "Kazakh", native: "Қазақша"),
        .init(code: "MN", name: "Mongolian", native: "Монгол"),
        .init(code: "TR", name: "Turkish", native: "Türkçe"),
        .init(code: "AR", name: "Arabic", native: "العربية"),
        .init(code: "FA", name: "Persian", native: "فارسی"),
        .init(code: "UR", name: "Urdu", native: "اردو"),
        .init(code: "UG", name: "Uyghur", native: "ئۇيغۇرچە"),
        .init(code: "HE", name: "Hebrew", native: "עברית"),
        .init(code: "HI", name: "Hindi", native: "हिन्दी"),
        .init(code: "BN", name: "Bengali", native: "বাংলা"),
        .init(code: "GU", name: "Gujarati", native: "ગુજરાતી"),
        .init(code: "MR", name: "Marathi", native: "मराठी"),
        .init(code: "TA", name: "Tamil", native: "தமிழ்"),
        .init(code: "TE", name: "Telugu", native: "తెలుగు"),
        .init(code: "BO", name: "Tibetan", native: "བོད་སྐད"),
        .init(code: "ZH", name: "Chinese (Simplified)", native: "简体中文"),
        .init(code: "ZT", name: "Chinese (Traditional)", native: "繁體中文"),
        .init(code: "YUE", name: "Cantonese", native: "粵語"),
        .init(code: "JA", name: "Japanese", native: "日本語"),
        .init(code: "KO", name: "Korean", native: "한국어"),
        .init(code: "VI", name: "Vietnamese", native: "Tiếng Việt"),
        .init(code: "TH", name: "Thai", native: "ไทย"),
        .init(code: "KM", name: "Khmer", native: "ខ្មែរ"),
        .init(code: "MY", name: "Burmese", native: "မြန်မာ"),
        .init(code: "ID", name: "Indonesian", native: "Bahasa Indonesia"),
        .init(code: "MS", name: "Malay", native: "Bahasa Melayu"),
        .init(code: "FIL", name: "Filipino", native: "Filipino"),
        .init(code: "YO", name: "Yoruba", native: "Yorùbá"),
        .init(code: "IG", name: "Igbo", native: "Igbo"),
        .init(code: "HA", name: "Hausa", native: "Hausa"),
        .init(code: "SW", name: "Swahili", native: "Kiswahili"),
        .init(code: "AM", name: "Amharic", native: "አማርኛ"),
        .init(code: "ZU", name: "Zulu", native: "isiZulu"),
    ]
    static func by(code: String) -> Language { all.first { $0.code == code } ?? all[0] }
}
