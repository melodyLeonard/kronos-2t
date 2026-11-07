import UIKit

/// The keyboard surface. Metrics follow the system keyboard on a 393pt iPhone: 33×42 keys, 6pt gaps, 12pt rows.
final class KeyboardView: UIView, UIInputViewAudioFeedback {
    enum Key {
        case char(String), space, backspace, ret, shift, capsLock, action
        case page(Page), emoji(String), pickLanguage(String), tone(String), sheet(Bool)
    }
    enum Page { case letters, numbers, symbols, emoji }
    enum Shift { case off, on, lock }
    enum State { case idle, ready, translating, translated }

    static let barHeight: CGFloat = 46
    static let rowsHeight: CGFloat = 216
    static var totalHeight: CGFloat { barHeight + rowsHeight }

    var onKey: ((Key) -> Void)?
    var globeHandler: (() -> Void)?
    var enableInputClicksWhenVisible: Bool { true }
    private(set) var shift: Shift = .off
    private var page: Page = .letters
    private var state: State = .idle
    private var sheetOpen = false
    private var needsGlobe: Bool

    private let bar = UIView()
    private let pill = UIButton(type: .system)
    private let status = UILabel()
    private let statusIcon = UIImageView()
    private let spinner = UIActivityIndicatorView(style: .medium)
    private let actionBtn = UIButton(type: .system)
    private let rows = UIStackView()
    private var sheet: UIView?
    private var flashTimer: Timer?

    init(needsGlobe: Bool) {
        self.needsGlobe = needsGlobe
        super.init(frame: .zero)
        build()
    }
    required init?(coder: NSCoder) { fatalError() }

    // MARK: palette

    private var dark: Bool { traitCollection.userInterfaceStyle == .dark }
    private var bed: UIColor { dark ? UIColor(white: 0.17, alpha: 1) : UIColor(red: 0.82, green: 0.827, blue: 0.851, alpha: 1) }
    private var keyFill: UIColor { dark ? UIColor(white: 0.353, alpha: 1) : .white }
    private var fnFill: UIColor { dark ? UIColor(white: 0.227, alpha: 1) : UIColor(red: 0.678, green: 0.702, blue: 0.737, alpha: 1) }
    private var ink: UIColor { dark ? .white : .black }
    private var keyShadow: UIColor { dark ? UIColor(white: 0.08, alpha: 1) : UIColor(red: 0.537, green: 0.541, blue: 0.553, alpha: 1) }
    private var blue: UIColor { UIColor(red: 0, green: 0.478, blue: 1, alpha: 1) }

    override func traitCollectionDidChange(_ previous: UITraitCollection?) {
        super.traitCollectionDidChange(previous)
        reload()
    }

    // MARK: build

    private func build() {
        backgroundColor = bed
        bar.translatesAutoresizingMaskIntoConstraints = false
        rows.translatesAutoresizingMaskIntoConstraints = false
        rows.axis = .vertical
        rows.spacing = 12
        rows.distribution = .fillEqually
        addSubview(bar); addSubview(rows)
        NSLayoutConstraint.activate([
            bar.topAnchor.constraint(equalTo: topAnchor, constant: 6),
            bar.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            bar.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            bar.heightAnchor.constraint(equalToConstant: 34),
            rows.topAnchor.constraint(equalTo: topAnchor, constant: Self.barHeight + 6),
            rows.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 3),
            rows.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -3),
            rows.heightAnchor.constraint(equalToConstant: 42 * 4 + 12 * 3),
        ])
        buildBar()
        buildRows()
    }

    private func buildBar() {
        for v in [pill, status, statusIcon, spinner, actionBtn] { v.translatesAutoresizingMaskIntoConstraints = false; bar.addSubview(v) }
        pill.addTarget(self, action: #selector(pillTap), for: .touchUpInside)
        actionBtn.addTarget(self, action: #selector(actionTap), for: .touchUpInside)
        status.font = .systemFont(ofSize: 13)
        status.adjustsFontSizeToFitWidth = true
        status.minimumScaleFactor = 0.85
        statusIcon.contentMode = .scaleAspectFit
        spinner.hidesWhenStopped = true
        NSLayoutConstraint.activate([
            pill.leadingAnchor.constraint(equalTo: bar.leadingAnchor),
            pill.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            pill.heightAnchor.constraint(equalToConstant: 34),
            actionBtn.trailingAnchor.constraint(equalTo: bar.trailingAnchor),
            actionBtn.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            actionBtn.heightAnchor.constraint(equalToConstant: 34),
            actionBtn.widthAnchor.constraint(greaterThanOrEqualToConstant: 44),
            statusIcon.leadingAnchor.constraint(equalTo: pill.trailingAnchor, constant: 8),
            statusIcon.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            statusIcon.widthAnchor.constraint(equalToConstant: 16),
            statusIcon.heightAnchor.constraint(equalToConstant: 16),
            spinner.centerXAnchor.constraint(equalTo: statusIcon.centerXAnchor),
            spinner.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            status.leadingAnchor.constraint(equalTo: statusIcon.trailingAnchor, constant: 5),
            status.trailingAnchor.constraint(lessThanOrEqualTo: actionBtn.leadingAnchor, constant: -8),
            status.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
        ])
    }

    private func buildRows() {
        rows.arrangedSubviews.forEach { $0.removeFromSuperview() }
        let spec: [[(String, Key, CGFloat, Bool)]]   // label/symbol, key, width, function key
        switch page {
        case .letters:
            spec = [
                "qwertyuiop".map { (String($0), .char(String($0)), 33, false) },
                "asdfghjkl".map { (String($0), .char(String($0)), 33, false) },
                [("sf:shift", .shift, 42, true)] + "zxcvbnm".map { (String($0), .char(String($0)), 33, false) } + [("sf:delete.left", .backspace, 42, true)],
                bottom(left: ("123", .page(.numbers), 44, true)),
            ]
        case .numbers:
            spec = [
                "1234567890".map { (String($0), .char(String($0)), 33, false) },
                ["-", "/", ":", ";", "(", ")", "$", "&", "@", "\""].map { ($0, .char($0), 33, false) },
                [("#+=", .page(.symbols), 42, true)] + [".", ",", "?", "!", "'"].map { ($0, .char($0), 44, false) } + [("sf:delete.left", .backspace, 42, true)],
                bottom(left: ("ABC", .page(.letters), 44, true)),
            ]
        case .symbols:
            spec = [
                ["[", "]", "{", "}", "#", "%", "^", "*", "+", "="].map { ($0, .char($0), 33, false) },
                ["_", "\\", "|", "~", "<", ">", "€", "£", "¥", "•"].map { ($0, .char($0), 33, false) },
                [("123", .page(.numbers), 42, true)] + [".", ",", "?", "!", "'"].map { ($0, .char($0), 44, false) } + [("sf:delete.left", .backspace, 42, true)],
                bottom(left: ("ABC", .page(.letters), 44, true)),
            ]
        case .emoji:
            let e = ["😀", "😂", "😍", "🥰", "😎", "🤔", "😅", "🙏", "👍", "👏", "🔥", "❤️", "🎉", "😢", "😡", "🤝", "💪", "✨", "🍝", "☕️"]
            spec = [
                e[0..<10].map { ($0, .emoji($0), 33, false) },
                e[10..<20].map { ($0, .emoji($0), 33, false) },
                [("sf:delete.left", .backspace, 42, true)],
                bottom(left: ("ABC", .page(.letters), 44, true)),
            ]
        }
        for line in spec {
            let h = UIStackView()
            h.axis = .horizontal; h.spacing = 6; h.alignment = .fill; h.distribution = .fill
            let lead = UIView(); let trail = UIView()
            lead.setContentHuggingPriority(.defaultLow, for: .horizontal)
            trail.setContentHuggingPriority(.defaultLow, for: .horizontal)
            h.addArrangedSubview(lead)
            for (label, key, w, fn) in line {
                let b = KeyButton(label: label, fn: fn, fill: fn ? fnFill : keyFill, ink: ink, shadow: keyShadow)
                b.widthAnchor.constraint(equalToConstant: w).isActive = true
                b.onTap = { [weak self] in self?.onKey?(key) }
                h.addArrangedSubview(b)
            }
            h.addArrangedSubview(trail)
            lead.widthAnchor.constraint(equalTo: trail.widthAnchor).isActive = true
            rows.addArrangedSubview(h)
        }
        applyShift()
        applyReturn()
    }

    private func bottom(left: (String, Key, CGFloat, Bool)) -> [(String, Key, CGFloat, Bool)] {
        var r: [(String, Key, CGFloat, Bool)] = [left]
        if needsGlobe { r.append(("sf:globe", .page(page), 42, true)) }
        r.append(("sf:face.smiling", .page(.emoji), 42, true))
        r.append(("space", .space, needsGlobe ? 134 : 182, false))
        r.append(("return", .ret, 90, true))
        return r
    }

    // MARK: state

    func reload() {
        backgroundColor = bed
        buildRows()
        let lang = Languages.by(code: Prefs.targetCode)
        var c = UIButton.Configuration.filled()
        c.baseBackgroundColor = dark ? UIColor(white: 0.353, alpha: 1) : .white
        c.baseForegroundColor = ink
        c.cornerStyle = .capsule
        c.contentInsets = .init(top: 0, leading: 11, bottom: 0, trailing: 9)
        let a = NSMutableAttributedString(string: "EN ", attributes: [.font: UIFont.systemFont(ofSize: 15, weight: .semibold)])
        a.append(icon("arrow.right", 11, .bold, dark ? UIColor(white: 0.7, alpha: 1) : .systemGray))
        a.append(NSAttributedString(string: " \(lang.code) ", attributes: [.font: UIFont.systemFont(ofSize: 15, weight: .semibold)]))
        a.append(icon("chevron.down", 9, .heavy, dark ? UIColor(white: 0.7, alpha: 1) : .systemGray))
        c.attributedTitle = AttributedString(a)
        pill.configuration = c
        pill.layer.shadowColor = keyShadow.cgColor; pill.layer.shadowOpacity = 1; pill.layer.shadowOffset = .init(width: 0, height: 1); pill.layer.shadowRadius = 0
        setState(state)
        if sheetOpen { setSheet(true) }
    }

    private func icon(_ name: String, _ size: CGFloat, _ weight: UIImage.SymbolWeight, _ color: UIColor) -> NSAttributedString {
        let img = UIImage(systemName: name, withConfiguration: UIImage.SymbolConfiguration(pointSize: size, weight: weight))?.withTintColor(color, renderingMode: .alwaysOriginal)
        let att = NSTextAttachment(); att.image = img
        if let img { att.bounds = CGRect(x: 0, y: -1, width: img.size.width, height: img.size.height) }
        return NSAttributedString(attachment: att)
    }

    func setState(_ s: State) {
        state = s
        flashTimer?.invalidate()
        let muted = dark ? UIColor(white: 0.82, alpha: 1) : UIColor(red: 0.235, green: 0.235, blue: 0.263, alpha: 1)
        status.textColor = muted
        spinner.stopAnimating()
        statusIcon.isHidden = false
        var c = UIButton.Configuration.filled()
        c.cornerStyle = .capsule
        c.contentInsets = .init(top: 0, leading: 12, bottom: 0, trailing: 12)
        c.imagePadding = 6
        c.imagePlacement = .leading
        var font = UIFont.systemFont(ofSize: 15, weight: .semibold)
        switch s {
        case .idle:
            statusIcon.isHidden = true
            status.text = "Type, then translate before you send"
            c.baseBackgroundColor = keyFill; c.baseForegroundColor = dark ? UIColor(white: 0.55, alpha: 1) : UIColor(white: 0.69, alpha: 1)
            c.image = UIImage(systemName: "translate")
        case .ready:
            statusIcon.image = UIImage(systemName: "bubble.left")?.withTintColor(muted, renderingMode: .alwaysOriginal)
            status.text = "\(Prefs.tone.capitalized) · context \(Prefs.contextOn ? "on" : "off")"
            c.baseBackgroundColor = blue; c.baseForegroundColor = .white
            c.image = UIImage(systemName: "translate"); c.title = "Translate"
        case .translating:
            statusIcon.isHidden = true; spinner.startAnimating()
            let host = URL(string: Prefs.serverURL)?.host ?? "your Mac"
            status.attributedText = bold("Translating", " on \(host)", muted)
            c.baseBackgroundColor = keyFill; c.baseForegroundColor = muted
            c.image = UIImage(systemName: "xmark", withConfiguration: UIImage.SymbolConfiguration(pointSize: 13, weight: .bold))
        case .translated:
            statusIcon.image = UIImage(systemName: "checkmark")?.withTintColor(.systemGreen, renderingMode: .alwaysOriginal)
            status.attributedText = bold("Translated", " · tap send", muted)
            c.baseBackgroundColor = keyFill; c.baseForegroundColor = blue
            c.image = UIImage(systemName: "arrow.uturn.backward"); c.title = "Undo"
        }
        c.titleTextAttributesTransformer = .init { var x = $0; x.font = font; return x }
        c.preferredSymbolConfigurationForImage = .init(pointSize: 15, weight: .semibold)
        font = .systemFont(ofSize: 15, weight: .semibold)
        actionBtn.configuration = c
        actionBtn.layer.shadowColor = (s == .ready ? UIColor(red: 0.04, green: 0.37, blue: 0.75, alpha: 1) : keyShadow).cgColor
        actionBtn.layer.shadowOpacity = 1; actionBtn.layer.shadowOffset = .init(width: 0, height: 1); actionBtn.layer.shadowRadius = 0
        applyReturn()
    }

    private func bold(_ b: String, _ rest: String, _ color: UIColor) -> NSAttributedString {
        let a = NSMutableAttributedString(string: b, attributes: [.font: UIFont.systemFont(ofSize: 13, weight: .semibold), .foregroundColor: dark ? UIColor.white : UIColor.black])
        a.append(NSAttributedString(string: rest, attributes: [.font: UIFont.systemFont(ofSize: 13), .foregroundColor: color]))
        return a
    }

    func flash(_ message: String) {
        statusIcon.image = UIImage(systemName: "exclamationmark.circle")?.withTintColor(.systemRed, renderingMode: .alwaysOriginal)
        statusIcon.isHidden = false
        status.text = message
        status.textColor = .systemRed
        flashTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: false) { [weak self] _ in
            guard let self else { return }
            self.setState(self.state)
        }
    }

    func setShift(_ s: Shift) { shift = s; applyShift() }
    private func applyShift() {
        for case let h as UIStackView in rows.arrangedSubviews {
            for case let b as KeyButton in h.arrangedSubviews {
                if b.label == "sf:shift" {
                    b.setSymbol(shift == .lock ? "capslock.fill" : (shift == .on ? "shift.fill" : "shift"))
                } else if !b.fn, b.label.count == 1, b.label.first!.isLetter {
                    b.setTitle(shift == .off ? b.label : b.label.uppercased())
                }
            }
        }
    }

    private func applyReturn() {
        for case let h as UIStackView in rows.arrangedSubviews {
            for case let b as KeyButton in h.arrangedSubviews where b.label == "return" {
                let send = state == .translated
                b.setTitle(send ? "send" : "return")
                b.restyle(fill: send ? blue : fnFill, ink: send ? .white : ink, shadow: send ? UIColor(red: 0.04, green: 0.37, blue: 0.75, alpha: 1) : keyShadow)
            }
        }
    }

    func setPage(_ p: Page) { page = p; buildRows() }

    func showGlobe(_ on: Bool) { if on != needsGlobe { needsGlobe = on; buildRows() } }

    // MARK: language sheet

    func setSheet(_ open: Bool) {
        sheetOpen = open
        sheet?.removeFromSuperview(); sheet = nil
        rows.isHidden = open
        bar.isHidden = open
        guard open else { return }
        let s = LanguageSheet(dark: dark) { [weak self] k in self?.onKey?(k) }
        s.translatesAutoresizingMaskIntoConstraints = false
        addSubview(s)
        NSLayoutConstraint.activate([
            s.topAnchor.constraint(equalTo: topAnchor), s.bottomAnchor.constraint(equalTo: bottomAnchor),
            s.leadingAnchor.constraint(equalTo: leadingAnchor), s.trailingAnchor.constraint(equalTo: trailingAnchor),
        ])
        sheet = s
    }

    @objc private func pillTap() { onKey?(.sheet(true)) }
    @objc private func actionTap() { onKey?(.action) }
}

/// One key cap: white (or function grey), 5pt radius, 1pt shadow, 24pt label or SF Symbol.
final class KeyButton: UIControl {
    let label: String
    let fn: Bool
    var onTap: (() -> Void)?
    private let text = UILabel()
    private let image = UIImageView()
    private var fill: UIColor
    private var pressedFill: UIColor

    init(label: String, fn: Bool, fill: UIColor, ink: UIColor, shadow: UIColor) {
        self.label = label; self.fn = fn; self.fill = fill; self.pressedFill = fill
        super.init(frame: .zero)
        layer.cornerRadius = 5
        layer.cornerCurve = .continuous
        layer.shadowOpacity = 1; layer.shadowOffset = .init(width: 0, height: 1); layer.shadowRadius = 0
        for v in [text, image] { v.translatesAutoresizingMaskIntoConstraints = false; v.isUserInteractionEnabled = false; addSubview(v) }
        text.textAlignment = .center
        image.contentMode = .center
        NSLayoutConstraint.activate([
            text.centerXAnchor.constraint(equalTo: centerXAnchor), text.centerYAnchor.constraint(equalTo: centerYAnchor),
            image.centerXAnchor.constraint(equalTo: centerXAnchor), image.centerYAnchor.constraint(equalTo: centerYAnchor),
        ])
        if label.hasPrefix("sf:") { setSymbol(String(label.dropFirst(3))) } else { setTitle(label) }
        restyle(fill: fill, ink: ink, shadow: shadow)
        addTarget(self, action: #selector(down), for: [.touchDown, .touchDragEnter])
        addTarget(self, action: #selector(up), for: [.touchUpInside, .touchUpOutside, .touchCancel, .touchDragExit])
        addTarget(self, action: #selector(tap), for: .touchUpInside)
    }
    required init?(coder: NSCoder) { fatalError() }

    func setTitle(_ t: String) {
        text.text = t
        let small = fn || t.count > 1
        text.font = .systemFont(ofSize: small ? 16 : 24, weight: .regular)
        image.isHidden = true; text.isHidden = false
    }
    func setSymbol(_ name: String) {
        image.image = UIImage(systemName: name, withConfiguration: UIImage.SymbolConfiguration(pointSize: 18, weight: .regular))
        image.isHidden = false; text.isHidden = true
    }
    func restyle(fill: UIColor, ink: UIColor, shadow: UIColor) {
        self.fill = fill
        backgroundColor = fill
        text.textColor = ink
        image.tintColor = ink
        layer.shadowColor = shadow.cgColor
    }
    @objc private func down() { backgroundColor = fn ? fill.withAlphaComponent(0.7) : fill.withAlphaComponent(0.85) }
    @objc private func up() { backgroundColor = fill }
    @objc private func tap() { onTap?() }
}

/// Language + tone picker that takes the place of the key rows.
final class LanguageSheet: UIView {
    init(dark: Bool, send: @escaping (KeyboardView.Key) -> Void) {
        super.init(frame: .zero)
        backgroundColor = dark ? UIColor(white: 0.17, alpha: 1) : UIColor(red: 0.82, green: 0.827, blue: 0.851, alpha: 1)
        let ink: UIColor = dark ? .white : .black
        let card: UIColor = dark ? UIColor(white: 0.353, alpha: 1) : .white
        let blue = UIColor(red: 0, green: 0.478, blue: 1, alpha: 1)

        let title = UILabel(); title.text = "Translate to"; title.font = .systemFont(ofSize: 17, weight: .semibold); title.textColor = ink
        let done = UIButton(type: .system); done.setTitle("Done", for: .normal); done.titleLabel?.font = .systemFont(ofSize: 17, weight: .semibold)
        done.addAction(UIAction { _ in send(.sheet(false)) }, for: .touchUpInside)
        let head = UIStackView(arrangedSubviews: [title, UIView(), done]); head.axis = .horizontal

        let chips = UIStackView(); chips.axis = .horizontal; chips.spacing = 8
        for code in Prefs.recents {
            let l = Languages.by(code: code)
            var c = UIButton.Configuration.filled(); c.cornerStyle = .capsule
            let on = code == Prefs.targetCode
            c.baseBackgroundColor = on ? blue : card; c.baseForegroundColor = on ? .white : ink
            c.contentInsets = .init(top: 0, leading: 10, bottom: 0, trailing: 12)
            let a = NSMutableAttributedString(string: l.code + "  ", attributes: [.font: UIFont.systemFont(ofSize: 11, weight: .bold)])
            a.append(NSAttributedString(string: l.name, attributes: [.font: UIFont.systemFont(ofSize: 15, weight: .semibold)]))
            c.attributedTitle = AttributedString(a)
            let b = UIButton(configuration: c); b.heightAnchor.constraint(equalToConstant: 34).isActive = true
            b.addAction(UIAction { _ in send(.pickLanguage(code)) }, for: .touchUpInside)
            chips.addArrangedSubview(b)
        }
        chips.addArrangedSubview(UIView())

        let seg = UISegmentedControl(items: ["Casual", "Neutral", "Formal"])
        seg.selectedSegmentIndex = ["casual", "neutral", "formal"].firstIndex(of: Prefs.tone) ?? 0
        seg.addAction(UIAction { _ in send(.tone(["casual", "neutral", "formal"][seg.selectedSegmentIndex])) }, for: .valueChanged)

        let list = UIStackView(); list.axis = .vertical
        list.backgroundColor = card; list.layer.cornerRadius = 10; list.clipsToBounds = true
        for (i, l) in Languages.all.enumerated() where l.code != "EN" {
            let row = UIButton(type: .system)
            var c = UIButton.Configuration.plain()
            c.contentInsets = .init(top: 0, leading: 14, bottom: 0, trailing: 14)
            c.baseForegroundColor = ink
            let a = NSMutableAttributedString(string: l.code + "   ", attributes: [.font: UIFont.systemFont(ofSize: 12, weight: .bold), .foregroundColor: dark ? UIColor(white: 0.75, alpha: 1) : UIColor.darkGray])
            a.append(NSAttributedString(string: l.name, attributes: [.font: UIFont.systemFont(ofSize: 17), .foregroundColor: ink]))
            c.attributedTitle = AttributedString(a)
            if l.code == Prefs.targetCode {
                c.image = UIImage(systemName: "checkmark", withConfiguration: UIImage.SymbolConfiguration(pointSize: 14, weight: .bold))
                c.imagePlacement = .trailing; c.baseForegroundColor = blue
            } else {
                c.subtitle = nil
            }
            c.titleAlignment = .leading
            row.configuration = c
            row.contentHorizontalAlignment = l.code == Prefs.targetCode ? .fill : .leading
            row.heightAnchor.constraint(equalToConstant: 44).isActive = true
            row.addAction(UIAction { _ in send(.pickLanguage(l.code)) }, for: .touchUpInside)
            if i > 0 {
                let sep = UIView(); sep.backgroundColor = dark ? UIColor(white: 0.45, alpha: 1) : UIColor(white: 0.9, alpha: 1)
                sep.heightAnchor.constraint(equalToConstant: 0.5).isActive = true
                list.addArrangedSubview(sep)
            }
            list.addArrangedSubview(row)
        }
        let scroll = UIScrollView()
        scroll.addSubview(list)
        list.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            list.topAnchor.constraint(equalTo: scroll.contentLayoutGuide.topAnchor),
            list.bottomAnchor.constraint(equalTo: scroll.contentLayoutGuide.bottomAnchor),
            list.leadingAnchor.constraint(equalTo: scroll.contentLayoutGuide.leadingAnchor),
            list.trailingAnchor.constraint(equalTo: scroll.contentLayoutGuide.trailingAnchor),
            list.widthAnchor.constraint(equalTo: scroll.frameLayoutGuide.widthAnchor),
        ])

        let v = UIStackView(arrangedSubviews: [head, chips, seg, scroll])
        v.axis = .vertical; v.spacing = 10
        v.translatesAutoresizingMaskIntoConstraints = false
        addSubview(v)
        NSLayoutConstraint.activate([
            v.topAnchor.constraint(equalTo: topAnchor, constant: 8),
            v.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            v.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            v.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -4),
        ])
    }
    required init?(coder: NSCoder) { fatalError() }
}
