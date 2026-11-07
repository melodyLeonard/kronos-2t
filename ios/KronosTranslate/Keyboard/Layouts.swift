import Foundation

/// Letter layouts per typing language, plus the accent variants shown on long press.
/// Scripts without a layout here (Indic, CJK, Thai…) fall back to QWERTY; those need an IME.
enum Layouts {
    struct Layout {
        let rows: [String]       // three letter rows
        let name: String
        var cased: Bool = true   // whether shift changes the glyphs
    }
    static let qwerty = Layout(rows: ["qwertyuiop", "asdfghjkl", "zxcvbnm"], name: "QWERTY")
    static let azerty = Layout(rows: ["azertyuiop", "qsdfghjklm", "wxcvbn'"], name: "AZERTY")
    static let qwertz = Layout(rows: ["qwertzuiop", "asdfghjkl", "yxcvbnm"], name: "QWERTZ")
    static let russian = Layout(rows: ["йцукенгшщзх", "фывапролджэ", "ячсмитьбю"], name: "ЙЦУКЕН")
    static let ukrainian = Layout(rows: ["йцукенгшщзх", "фівапролджє", "ячсмитьбю"], name: "ЙЦУКЕН")
    static let greek = Layout(rows: ["ςερτυθιοπ", "ασδφγηξκλ", "ζχψωβνμ"], name: "Greek")
    static let arabic = Layout(rows: ["ضصثقفغعهخحج", "شسيبلاتنمك", "ئءؤرىةوزظد"], name: "Arabic", cased: false)
    static let persian = Layout(rows: ["ضصثقفغعهخحج", "شسیبلاتنمکگ", "ظطزرذدپو"], name: "Persian", cased: false)
    static let hebrew = Layout(rows: ["קראטוןםפ", "שדגכעיחלךף", "זסבהנמצתץ"], name: "Hebrew", cased: false)

    static func layout(for code: String) -> Layout {
        switch code {
        case "FR": return azerty
        case "DE", "CS", "SK", "HU": return qwertz
        case "RU", "KK", "MN": return russian
        case "UK": return ukrainian
        case "EL": return greek
        case "AR", "UG": return arabic
        case "FA", "UR": return persian
        case "HE": return hebrew
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
