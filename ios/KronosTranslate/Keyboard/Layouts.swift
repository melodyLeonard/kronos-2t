import Foundation

/// Letter layouts per typing language, plus the accent variants shown on long press.
enum Layouts {
    struct Layout {
        let rows: [String]       // three letter rows
        let name: String
    }
    static let qwerty = Layout(rows: ["qwertyuiop", "asdfghjkl", "zxcvbnm"], name: "QWERTY")
    static let azerty = Layout(rows: ["azertyuiop", "qsdfghjklm", "wxcvbn'"], name: "AZERTY")
    static let qwertz = Layout(rows: ["qwertzuiop", "asdfghjkl", "yxcvbnm"], name: "QWERTZ")

    static func layout(for code: String) -> Layout {
        switch code {
        case "FR": return azerty
        case "DE", "CS", "PL": return qwertz
        default: return qwerty
        }
    }

    static let accents: [String: [String]] = [
        "a": ["à", "á", "â", "ä", "æ", "ã", "å", "ā", "ą"],
        "e": ["è", "é", "ê", "ë", "ē", "ė", "ę", "ẹ"],
        "i": ["î", "ï", "í", "ī", "į", "ì", "ị"],
        "o": ["ô", "ö", "ò", "ó", "œ", "ø", "ō", "õ", "ọ"],
        "u": ["û", "ü", "ù", "ú", "ū", "ų"],
        "c": ["ç", "ć", "č"],
        "n": ["ñ", "ń", "ň"],
        "s": ["ß", "ś", "š", "ṣ"],
        "y": ["ÿ", "ý"],
        "z": ["ž", "ź", "ż"],
        "l": ["ł"],
        "g": ["ğ"],
        "'": ["‘", "’", "‚", "‛", "«", "»"],
    ]
}
