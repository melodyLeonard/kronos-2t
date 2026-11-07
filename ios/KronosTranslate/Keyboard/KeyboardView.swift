import UIKit

/// The keyboard surface. Key sizes derive from the width the way the system keyboard's do
/// (ten letter keys, 6pt gaps, 3pt margins, 42pt rows, 12pt row gaps).
final class KeyboardView: UIView, UIInputViewAudioFeedback {
    enum Key {
        case char(String), space, backspace, ret, shift, capsLock, action, globe
        case page(Page), emoji(String), pickLanguage(String), pickSource(String), sheet(Bool)
    }
    enum Page { case letters, numbers, symbols, emoji }
    enum Shift { case off, on, lock }
    /// `preview` carries the live translation of the field; `inserted` means it replaced the field.
    enum State: Equatable { case idle, translating, preview(String), inserted(String) }

    static let stripHeight: CGFloat = 46
    static let suggestHeight: CGFloat = 44
    static let rowsHeight: CGFloat = 212
    /// Strip only while there is something to show; suggestions row always, like the system keyboard.
    var height: CGFloat { (stripVisible ? Self.stripHeight : 0) + Self.suggestHeight + Self.rowsHeight }
    private var stripVisible: Bool { state != .idle && !sheetOpen && page != .emoji }

    var onKey: ((Key) -> Void)?
    var onSuggestion: ((String) -> Void)?
    var onHeightChange: ((CGFloat) -> Void)?
    var enableInputClicksWhenVisible: Bool { true }
    private(set) var shift: Shift = .off
    private var page: Page = .letters
    private(set) var state: State = .idle
    private var sheetOpen = false
    private var needsGlobe: Bool

    private let bar = UIView()          // reply-style reference strip: accent | title + translation | action
    private let accent = UIView()
    private let pill = UIButton(type: .custom)
    private let status = UILabel()
    private let spinner = UIActivityIndicatorView(style: .medium)
    private let actionBtn = UIButton(type: .custom)
    private let keysView = UIView()
    private let suggestBar = UIView()
    private var suggestButtons: [UIButton] = []
    private var suggestDividers: [UIView] = []
    private var rows: [[(KeyButton, CGFloat)]] = []   // button + width in letter units (0 = flex)
    private var sheet: UIView?
    private var emojiView: EmojiKeyboard?
    private var popup: AccentPopup?
    private var flashTimer: Timer?

    init(needsGlobe: Bool) {
        self.needsGlobe = needsGlobe
        super.init(frame: .zero)
        build()
    }
    required init?(coder: NSCoder) { fatalError() }

    // MARK: palette

    var dark: Bool { traitCollection.userInterfaceStyle == .dark }
    private var bed: UIColor { dark ? UIColor(white: 0.17, alpha: 1) : UIColor(red: 0.82, green: 0.827, blue: 0.851, alpha: 1) }
    private var keyFill: UIColor { dark ? UIColor(white: 0.353, alpha: 1) : .white }
    // Function keys on iOS 26 are a translucent tint of the backdrop, not solid grey.
    private var fnFill: UIColor { dark ? UIColor(white: 1, alpha: 0.12) : UIColor(red: 0.86, green: 0.87, blue: 0.89, alpha: 0.9) }
    private var ink: UIColor { dark ? .white : .black }
    private var muted: UIColor { dark ? UIColor(white: 0.82, alpha: 1) : UIColor(red: 0.235, green: 0.235, blue: 0.263, alpha: 1) }
    private var keyShadow: UIColor { dark ? UIColor(white: 0, alpha: 0.5) : UIColor(white: 0, alpha: 0.22) }
    private var blue: UIColor { UIColor(red: 0, green: 0.478, blue: 1, alpha: 1) }
    private var blueShadow: UIColor { UIColor(red: 0.04, green: 0.37, blue: 0.75, alpha: 1) }

    override func traitCollectionDidChange(_ previous: UITraitCollection?) {
        super.traitCollectionDidChange(previous)
        if previous?.userInterfaceStyle != traitCollection.userInterfaceStyle { reload() }
    }

    // MARK: build

    private func build() {
        // Clear: the system draws its keyboard backdrop (blur/material) behind extensions.
        backgroundColor = .clear
        addSubview(bar)
        addSubview(suggestBar)
        addSubview(keysView)
        buildBar()
        buildSuggestBar()
        buildRows()
    }

    private func buildSuggestBar() {
        for i in 0..<3 {
            let b = UIButton(type: .custom)
            b.titleLabel?.font = .systemFont(ofSize: 17)
            b.titleLabel?.lineBreakMode = .byTruncatingTail
            b.tag = i
            b.addTarget(self, action: #selector(suggestTap(_:)), for: .touchUpInside)
            suggestBar.addSubview(b)
            suggestButtons.append(b)
            if i < 2 {
                let d = UIView()
                suggestBar.addSubview(d)
                suggestDividers.append(d)
            }
        }
    }

    func setSuggestions(_ s: [String]) {
        for (i, b) in suggestButtons.enumerated() {
            let t = i < s.count ? s[i] : ""
            b.setTitle(t, for: .normal)
            b.setTitleColor(ink, for: .normal)
            b.isHidden = t.isEmpty
        }
        for (i, d) in suggestDividers.enumerated() {
            d.backgroundColor = dark ? UIColor(white: 1, alpha: 0.18) : UIColor(white: 0, alpha: 0.16)
            d.isHidden = i + 1 >= s.count
        }
    }

    @objc private func suggestTap(_ b: UIButton) {
        guard let t = b.title(for: .normal), !t.isEmpty else { return }
        onSuggestion?(t.hasPrefix("\u{201C}") ? String(t.dropFirst().dropLast()) : t)
    }

    private func buildBar() {
        bar.layer.cornerRadius = 10
        bar.layer.cornerCurve = .continuous
        bar.clipsToBounds = true
        for v in [accent, pill, status, spinner, actionBtn] { v.translatesAutoresizingMaskIntoConstraints = false; bar.addSubview(v) }
        pill.isUserInteractionEnabled = false
        bar.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(barTap(_:))))
        actionBtn.addTarget(self, action: #selector(actionTap), for: .touchUpInside)
        status.font = .systemFont(ofSize: 15)
        status.lineBreakMode = .byTruncatingTail
        status.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        status.setContentHuggingPriority(.defaultLow, for: .horizontal)
        pill.contentHorizontalAlignment = .leading
        actionBtn.setContentCompressionResistancePriority(.required, for: .horizontal)
        actionBtn.setContentHuggingPriority(.required, for: .horizontal)
        spinner.hidesWhenStopped = true
        spinner.transform = CGAffineTransform(scaleX: 0.7, y: 0.7)
        NSLayoutConstraint.activate([
            accent.leadingAnchor.constraint(equalTo: bar.leadingAnchor),
            accent.topAnchor.constraint(equalTo: bar.topAnchor),
            accent.bottomAnchor.constraint(equalTo: bar.bottomAnchor),
            accent.widthAnchor.constraint(equalToConstant: 4),
            pill.leadingAnchor.constraint(equalTo: accent.trailingAnchor, constant: 8),
            pill.topAnchor.constraint(equalTo: bar.topAnchor, constant: 2),
            pill.heightAnchor.constraint(equalToConstant: 18),
            pill.trailingAnchor.constraint(lessThanOrEqualTo: actionBtn.leadingAnchor, constant: -8),
            spinner.leadingAnchor.constraint(equalTo: pill.trailingAnchor, constant: 4),
            spinner.centerYAnchor.constraint(equalTo: pill.centerYAnchor),
            status.leadingAnchor.constraint(equalTo: accent.trailingAnchor, constant: 8),
            status.trailingAnchor.constraint(lessThanOrEqualTo: actionBtn.leadingAnchor, constant: -8),
            status.topAnchor.constraint(equalTo: pill.bottomAnchor, constant: 1),
            actionBtn.trailingAnchor.constraint(equalTo: bar.trailingAnchor, constant: -6),
            actionBtn.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            actionBtn.heightAnchor.constraint(equalToConstant: 34),
            actionBtn.widthAnchor.constraint(greaterThanOrEqualToConstant: 44),
        ])
    }

    private func key(_ label: String, _ k: Key, fn: Bool = false) -> KeyButton {
        let b = KeyButton(label: label, fn: fn, fill: fn ? fnFill : keyFill, ink: ink, shadow: keyShadow)
        b.onTap = { [weak self] in self?.onKey?(k) }
        if case .char(let c) = k, let v = Layouts.accents[c] {
            b.variants = v
            b.onLongPress = { [weak self, weak b] g in if let b { self?.accent(g, on: b) } }
        }
        return b
    }

    private func buildRows() {
        rows.flatMap { $0 }.forEach { $0.0.removeFromSuperview() }
        rows = []
        let fn = 1.27, big = 1.33
        func letters(_ s: String) -> [(KeyButton, CGFloat)] { s.map { (key(String($0), .char(String($0))), 1) } }
        switch page {
        case .letters:
            let l = Layouts.layout(for: Prefs.sourceCode)
            rows.append(letters(l.rows[0]))
            rows.append(letters(l.rows[1]))
            let shiftKey: [(KeyButton, CGFloat)] = l.cased ? [(key("sf:shift", .shift, fn: true), fn)] : []
            rows.append(shiftKey + letters(l.rows[2]) + [(key("sf:delete.left", .backspace, fn: true), fn)])
            rows.append(bottom(left: (key("123", .page(.numbers), fn: true), big)))
        case .numbers:
            rows.append(letters("1234567890"))
            rows.append(["-", "/", ":", ";", "(", ")", "$", "&", "@", "\""].map { (key($0, .char($0)), 1) })
            rows.append([(key("#+=", .page(.symbols), fn: true), fn)] + [".", ",", "?", "!", "'"].map { (key($0, .char($0)), 0) } + [(key("sf:delete.left", .backspace, fn: true), fn)])
            rows.append(bottom(left: (key("ABC", .page(.letters), fn: true), big)))
        case .symbols:
            rows.append(["[", "]", "{", "}", "#", "%", "^", "*", "+", "="].map { (key($0, .char($0)), 1) })
            rows.append(["_", "\\", "|", "~", "<", ">", "€", "£", "¥", "•"].map { (key($0, .char($0)), 1) })
            rows.append([(key("123", .page(.numbers), fn: true), fn)] + [".", ",", "?", "!", "'"].map { (key($0, .char($0)), 0) } + [(key("sf:delete.left", .backspace, fn: true), fn)])
            rows.append(bottom(left: (key("ABC", .page(.letters), fn: true), big)))
        case .emoji:
            break
        }
        for r in rows { for (b, _) in r { keysView.addSubview(b) } }
        applyShift()
        applyReturn()
        setNeedsLayout()
    }

    private func bottom(left: (KeyButton, CGFloat)) -> [(KeyButton, CGFloat)] {
        var r = [left]
        if needsGlobe { r.append((key("sf:globe", .globe, fn: true), 1.27)) }
        r.append((key("sf:face.smiling", .page(.emoji), fn: true), 1.27))
        r.append((key(" ", .space), 0))
        r.append((key("sf:return", .ret, fn: true), 2.73))
        return r
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        let W = bounds.width
        var y: CGFloat = 0
        bar.isHidden = !stripVisible
        if stripVisible {
            bar.frame = CGRect(x: 8, y: 2, width: W - 16, height: 44)
            y = Self.stripHeight
        }
        suggestBar.frame = CGRect(x: 0, y: y, width: W, height: Self.suggestHeight)
        let third = W / 3
        for (i, b) in suggestButtons.enumerated() { b.frame = CGRect(x: third * CGFloat(i) + 6, y: 0, width: third - 12, height: Self.suggestHeight) }
        for (i, d) in suggestDividers.enumerated() { d.frame = CGRect(x: third * CGFloat(i + 1) - 0.5, y: 10, width: 1, height: Self.suggestHeight - 20) }
        y += Self.suggestHeight
        keysView.frame = CGRect(x: 0, y: y, width: W, height: Self.rowsHeight)
        emojiView?.frame = bounds
        sheet?.frame = bounds
        let gap: CGFloat = 6, margin: CGFloat = 3
        // Letter width from the widest letter row (10 on QWERTY, 11 on Cyrillic/Arabic).
        let widest = max(10, rows.prefix(2).map { $0.filter { $0.1 == 1 }.count }.max() ?? 10)
        let unit = floor((W - 2 * margin - CGFloat(widest - 1) * gap) / CGFloat(widest))
        let rowH: CGFloat = 40, rowGap: CGFloat = 13   // measured from the system keyboard on iOS 26
        for (i, r) in rows.enumerated() {
            var fixed = r.map { $0.1 * unit }.reduce(0, +)
            let flexCount = r.filter { $0.1 == 0 }.count
            let avail = W - 2 * margin - gap * CGFloat(r.count - 1)
            // A row wider than the screen (11 letters + delete) shrinks evenly.
            let squeeze = flexCount == 0 && fixed > avail ? avail / fixed : 1
            fixed *= squeeze
            let flexW = flexCount > 0 ? (avail - fixed) / CGFloat(flexCount) : 0
            let total = fixed + flexW * CGFloat(flexCount) + gap * CGFloat(r.count - 1)
            var x = (W - total) / 2
            let y = CGFloat(i) * (rowH + rowGap)
            for (b, u) in r {
                let w = u == 0 ? flexW : u * unit * squeeze
                b.frame = CGRect(x: x, y: y, width: floor(w), height: rowH)
                x += floor(w) + gap
            }
        }
    }

    // MARK: accents

    private func accent(_ g: UILongPressGestureRecognizer, on key: KeyButton) {
        switch g.state {
        case .began:
            let p = AccentPopup(variants: key.variants, dark: dark)
            let frame = key.convert(key.bounds, to: self)
            p.place(above: frame, in: bounds)
            addSubview(p)
            popup = p
            key.setPressed(false)
        case .changed:
            popup?.track(g.location(in: self))
        case .ended:
            if let v = popup?.selected { onKey?(.char(v)) }
            popup?.removeFromSuperview(); popup = nil
        default:
            popup?.removeFromSuperview(); popup = nil
        }
    }

    // MARK: state

    func reload() {
        backgroundColor = .clear
        bar.backgroundColor = .clear   // sits straight on the backdrop like the suggestions row
        setSuggestions([])
        buildRows()
        var c = UIButton.Configuration.plain()
        c.contentInsets = .zero
        c.titleLineBreakMode = .byClipping
        let gray = dark ? UIColor(white: 0.7, alpha: 1) : UIColor.systemGray
        let f = UIFont.systemFont(ofSize: 12, weight: .semibold)
        let a = NSMutableAttributedString(string: Prefs.sourceCode + " ", attributes: [.font: f, .foregroundColor: blue])
        a.append(icon("arrow.right", 9, .bold, blue))
        a.append(NSAttributedString(string: " \(Languages.by(code: Prefs.targetCode).name) ", attributes: [.font: f, .foregroundColor: blue]))
        a.append(icon("chevron.down", 8, .heavy, blue))
        _ = gray
        c.attributedTitle = AttributedString(a)
        pill.configuration = c
        setState(state)
    }

    private func shadow(_ v: UIView, _ color: UIColor) {
        v.layer.shadowColor = color.cgColor
        v.layer.shadowOpacity = 1
        v.layer.shadowOffset = .init(width: 0, height: 1)
        v.layer.shadowRadius = 0
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
        status.textColor = ink
        status.font = .systemFont(ofSize: 15)
        accent.backgroundColor = blue
        spinner.stopAnimating()
        var c = UIButton.Configuration.filled()
        c.cornerStyle = .capsule
        c.contentInsets = .init(top: 0, leading: 12, bottom: 0, trailing: 12)
        c.imagePadding = 5
        c.imagePlacement = .leading
        c.titleLineBreakMode = .byClipping
        c.preferredSymbolConfigurationForImage = .init(pointSize: 14, weight: .semibold)
        actionBtn.isHidden = false
        switch s {
        case .idle:
            actionBtn.isHidden = true   // strip is hidden entirely
        case .translating:
            spinner.startAnimating()
            if case .preview(let old) = state { status.text = old } else { status.text = "…" }
            status.textColor = dark ? UIColor(white: 0.7, alpha: 1) : UIColor(white: 0.5, alpha: 1)
            actionBtn.isHidden = true
        case .preview(let t):
            status.text = t
            c.baseBackgroundColor = blue; c.baseForegroundColor = .white
            c.image = UIImage(systemName: "arrow.up.doc.on.clipboard"); c.title = "Use"
        case .inserted(let original):
            status.text = original
            status.textColor = dark ? UIColor(white: 0.7, alpha: 1) : UIColor(white: 0.5, alpha: 1)
            accent.backgroundColor = .systemGreen
            c.baseBackgroundColor = dark ? UIColor(white: 0.25, alpha: 1) : UIColor(white: 0.93, alpha: 1); c.baseForegroundColor = blue
            c.image = UIImage(systemName: "arrow.uturn.backward"); c.title = "Undo"
        }
        c.titleTextAttributesTransformer = .init { var x = $0; x.font = UIFont.systemFont(ofSize: 14, weight: .semibold); return x }
        actionBtn.configuration = c
        applyReturn()
        setNeedsLayout()
        onHeightChange?(height)
    }

    /// Keeps the previous preview visible while a newer one is in flight.
    func setTranslating() {
        let keep = state
        if case .translating = keep { return }   // already spinning; nothing to redraw
        state = .translating
        setState(.translating)
        if case .preview(let old) = keep { status.text = old }
    }

    func flash(_ message: String) {
        status.text = message
        status.textColor = .systemRed
        accent.backgroundColor = .systemRed
        flashTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: false) { [weak self] _ in
            guard let self else { return }
            self.setState(self.state)
        }
    }

    func setShift(_ s: Shift) { shift = s; applyShift() }
    private func applyShift() {
        for (b, _) in rows.flatMap({ $0 }) {
            if b.label == "sf:shift" {
                b.setSymbol(shift == .lock ? "capslock.fill" : (shift == .on ? "shift.fill" : "shift"))
            } else if !b.fn, b.label.count == 1, b.label.first!.isLetter, Layouts.layout(for: Prefs.sourceCode).cased {
                b.setTitle(shift == .off ? b.label : b.label.uppercased())
            }
        }
    }

    private func applyReturn() {
        for (b, _) in rows.flatMap({ $0 }) where b.label == "sf:return" {
            var send = false
            if case .inserted = state { send = true }
            if send { b.setTitle("send") } else { b.setSymbol("return") }
            b.restyle(fill: send ? blue : fnFill, ink: send ? .white : ink, shadow: send ? blueShadow : keyShadow)
        }
    }

    func setPage(_ p: Page) {
        page = p
        emojiView?.removeFromSuperview(); emojiView = nil
        let emoji = p == .emoji
        keysView.isHidden = emoji
        suggestBar.isHidden = emoji
        if emoji {
            let e = EmojiKeyboard(dark: dark)
            e.onEmoji = { [weak self] s in self?.onKey?(.emoji(s)) }
            e.onABC = { [weak self] in self?.onKey?(.page(.letters)) }
            e.onBackspace = { [weak self] in self?.onKey?(.backspace) }
            addSubview(e)
            emojiView = e
        } else {
            buildRows()
        }
        setNeedsLayout()
        onHeightChange?(height)
    }

    func showGlobe(_ on: Bool) { if on != needsGlobe { needsGlobe = on; buildRows() } }

    // MARK: language sheet

    func setSheet(_ open: Bool, side: LanguageSheet.Side = .to) {
        sheetOpen = open
        sheet?.removeFromSuperview(); sheet = nil
        keysView.isHidden = open
        suggestBar.isHidden = open
        setNeedsLayout()
        onHeightChange?(height)
        guard open else { return }
        let s = LanguageSheet(dark: dark, side: side) { [weak self] k in self?.onKey?(k) }
        addSubview(s)
        sheet = s
    }

    /// Anywhere on the strip opens the picker: the source code on the far left, the target elsewhere.
    @objc private func barTap(_ g: UITapGestureRecognizer) {
        let p = g.location(in: bar)
        if actionBtn.frame.contains(p), !actionBtn.isHidden { return }
        setSheet(true, side: p.x < 40 ? .from : .to)
    }
    @objc private func actionTap() { onKey?(.action) }
}

/// One key cap: white (or function grey), 5pt radius, 1pt shadow, 24pt label or SF Symbol.
final class KeyButton: UIControl {
    let label: String
    let fn: Bool
    var variants: [String] = []
    var onTap: (() -> Void)?
    var onLongPress: ((UILongPressGestureRecognizer) -> Void)?
    private let text = UILabel()
    private let image = UIImageView()
    private var fill: UIColor = .white
    private var pressing = false

    init(label: String, fn: Bool, fill: UIColor, ink: UIColor, shadow: UIColor) {
        self.label = label; self.fn = fn
        super.init(frame: .zero)
        layer.cornerRadius = 10
        layer.cornerCurve = .continuous
        // Soft elevation like the system keycaps, not a hard 1pt line.
        layer.shadowOpacity = 0   // flat, like the iOS 26 keycaps
        for v in [text, image] { v.translatesAutoresizingMaskIntoConstraints = false; v.isUserInteractionEnabled = false; addSubview(v) }
        text.textAlignment = .center
        image.contentMode = .center
        NSLayoutConstraint.activate([
            text.centerXAnchor.constraint(equalTo: centerXAnchor), text.centerYAnchor.constraint(equalTo: centerYAnchor),
            image.centerXAnchor.constraint(equalTo: centerXAnchor), image.centerYAnchor.constraint(equalTo: centerYAnchor),
        ])
        if label.hasPrefix("sf:") { setSymbol(String(label.dropFirst(3))) } else { setTitle(label) }
        restyle(fill: fill, ink: ink, shadow: shadow)
        let lp = UILongPressGestureRecognizer(target: self, action: #selector(longPress(_:)))
        lp.minimumPressDuration = 0.35
        addGestureRecognizer(lp)
    }
    required init?(coder: NSCoder) { fatalError() }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) { super.touchesBegan(touches, with: event); setPressed(true) }
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesEnded(touches, with: event)
        setPressed(false)
        if let t = touches.first, bounds.insetBy(dx: -8, dy: -8).contains(t.location(in: self)), !longPressed { onTap?() }
        longPressed = false
    }
    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) { super.touchesCancelled(touches, with: event); setPressed(false); longPressed = false }
    private var longPressed = false
    @objc private func longPress(_ g: UILongPressGestureRecognizer) {
        guard !variants.isEmpty else { return }
        if g.state == .began { longPressed = true }
        onLongPress?(g)
    }

    func setPressed(_ on: Bool) { backgroundColor = on ? fill.withAlphaComponent(fn ? 0.7 : 0.85) : fill }
    func setTitle(_ t: String) {
        text.text = t
        // Apple's keycaps: SF Pro, light weight for single glyphs, regular 16pt for words.
        let small = fn || t.count > 1
        text.font = small ? .systemFont(ofSize: 16, weight: .regular) : .systemFont(ofSize: 22, weight: .regular)
        image.isHidden = true; text.isHidden = false
    }
    func setSymbol(_ name: String) {
        image.image = UIImage(systemName: name, withConfiguration: UIImage.SymbolConfiguration(pointSize: 17, weight: .regular))
        image.isHidden = false; text.isHidden = true
    }
    func restyle(fill: UIColor, ink: UIColor, shadow: UIColor) {
        self.fill = fill
        backgroundColor = fill
        text.textColor = ink
        image.tintColor = ink
        layer.shadowColor = shadow.cgColor
    }
}

/// The accent chooser shown above a key on long press; slide to pick, lift to insert.
final class AccentPopup: UIView {
    private let variants: [String]
    private var labels: [UILabel] = []
    private(set) var selected: String?
    private let cellW: CGFloat = 34
    private let blue = UIColor(red: 0, green: 0.478, blue: 1, alpha: 1)

    init(variants: [String], dark: Bool) {
        self.variants = variants
        super.init(frame: .zero)
        backgroundColor = dark ? UIColor(white: 0.35, alpha: 1) : .white
        layer.cornerRadius = 8
        layer.cornerCurve = .continuous
        layer.shadowColor = UIColor.black.cgColor; layer.shadowOpacity = 0.3; layer.shadowRadius = 6; layer.shadowOffset = .init(width: 0, height: 2)
        for v in variants {
            let l = UILabel()
            l.text = v; l.font = .systemFont(ofSize: 24); l.textAlignment = .center
            l.textColor = dark ? .white : .black
            l.layer.cornerRadius = 6; l.clipsToBounds = true
            addSubview(l); labels.append(l)
        }
        select(0)
    }
    required init?(coder: NSCoder) { fatalError() }

    func place(above key: CGRect, in bounds: CGRect) {
        let w = cellW * CGFloat(variants.count) + 8, h: CGFloat = 48
        var x = key.midX - w / 2
        x = max(4, min(bounds.width - w - 4, x))
        frame = CGRect(x: x, y: key.minY - h - 6, width: w, height: h)
        for (i, l) in labels.enumerated() { l.frame = CGRect(x: 4 + CGFloat(i) * cellW, y: 4, width: cellW, height: h - 8) }
    }
    func track(_ p: CGPoint) {
        let local = convert(p, from: superview)
        let i = Int((local.x - 4) / cellW)
        select(max(0, min(variants.count - 1, i)))
    }
    private func select(_ i: Int) {
        selected = variants[i]
        for (j, l) in labels.enumerated() {
            l.backgroundColor = j == i ? blue : .clear
            l.textColor = j == i ? .white : (backgroundColor == .white ? .black : .white)
        }
    }
}

/// Language picker: the system wheel, two columns (typing language | receiving language).
final class LanguageSheet: UIView, UIPickerViewDataSource, UIPickerViewDelegate {
    enum Side { case from, to }
    private let send: (KeyboardView.Key) -> Void
    private let wheel = UIPickerView()
    private let dark: Bool
    private let langs = Languages.all

    init(dark: Bool, side: Side, send: @escaping (KeyboardView.Key) -> Void) {
        self.send = send
        self.dark = dark
        super.init(frame: .zero)
        backgroundColor = dark ? UIColor(white: 0.17, alpha: 1) : UIColor(red: 0.82, green: 0.827, blue: 0.851, alpha: 1)
        let ink: UIColor = dark ? .white : .black

        func header(_ s: String) -> UILabel {
            let l = UILabel(); l.text = s.uppercased(); l.textAlignment = .center
            l.font = .systemFont(ofSize: 11, weight: .semibold); l.textColor = .systemGray
            return l
        }
        let done = UIButton(type: .system)
        done.setTitle("Done", for: .normal)
        done.titleLabel?.font = .systemFont(ofSize: 17, weight: .semibold)
        done.addAction(UIAction { [weak self] _ in self?.commit(); send(.sheet(false)) }, for: .touchUpInside)
        let heads = UIStackView(arrangedSubviews: [header("I type in"), header("They receive in")])
        heads.axis = .horizontal; heads.distribution = .fillEqually

        wheel.dataSource = self
        wheel.delegate = self
        wheel.overrideUserInterfaceStyle = dark ? .dark : .light
        let from = langs.firstIndex { $0.code == Prefs.sourceCode } ?? 0
        let to = langs.firstIndex { $0.code == Prefs.targetCode } ?? 1
        wheel.selectRow(from, inComponent: 0, animated: false)
        wheel.selectRow(to, inComponent: 1, animated: false)

        for v in [heads, wheel, done] { v.translatesAutoresizingMaskIntoConstraints = false; addSubview(v) }
        NSLayoutConstraint.activate([
            heads.topAnchor.constraint(equalTo: topAnchor, constant: 10),
            heads.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            heads.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            heads.heightAnchor.constraint(equalToConstant: 16),
            done.centerYAnchor.constraint(equalTo: wheel.centerYAnchor),
            done.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -12),
            done.widthAnchor.constraint(equalToConstant: 52),
            wheel.topAnchor.constraint(equalTo: heads.bottomAnchor, constant: -4),
            wheel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            wheel.trailingAnchor.constraint(equalTo: done.leadingAnchor, constant: -4),
            wheel.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -2),
            heads.trailingAnchor.constraint(equalTo: wheel.trailingAnchor),
        ])
        _ = ink
    }
    required init?(coder: NSCoder) { fatalError() }

    /// Writes both wheels back; the controller reloads the layout and preview.
    private func commit() {
        let from = langs[wheel.selectedRow(inComponent: 0)].code
        let to = langs[wheel.selectedRow(inComponent: 1)].code
        if from != Prefs.sourceCode { send(.pickSource(from)) }
        if to != Prefs.targetCode { send(.pickLanguage(to)) }
    }

    func numberOfComponents(in pickerView: UIPickerView) -> Int { 2 }
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int { langs.count }
    func pickerView(_ pickerView: UIPickerView, rowHeightForComponent component: Int) -> CGFloat { 34 }
    func pickerView(_ pickerView: UIPickerView, viewForRow row: Int, forComponent component: Int, reusing view: UIView?) -> UIView {
        let l = (view as? UILabel) ?? UILabel()
        let lang = langs[row]
        let a = NSMutableAttributedString(string: lang.name, attributes: [.font: UIFont.systemFont(ofSize: 17), .foregroundColor: dark ? UIColor.white : UIColor.black])
        if lang.native != lang.name {
            a.append(NSAttributedString(string: "  " + lang.native, attributes: [.font: UIFont.systemFont(ofSize: 13), .foregroundColor: UIColor.systemGray]))
        }
        l.attributedText = a
        l.textAlignment = .center
        l.adjustsFontSizeToFitWidth = true
        l.minimumScaleFactor = 0.8
        return l
    }
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        // Same language on both sides: nudge the other wheel to what this one was.
        let other = component == 0 ? 1 : 0
        if wheel.selectedRow(inComponent: other) == row {
            let prev = langs.firstIndex { $0.code == (component == 0 ? Prefs.sourceCode : Prefs.targetCode) } ?? 0
            wheel.selectRow(prev, inComponent: other, animated: true)
        }
        commit()
    }
}
