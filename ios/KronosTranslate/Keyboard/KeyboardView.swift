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

    static let barHeight: CGFloat = 60
    static let rowsHeight: CGFloat = 216
    static var totalHeight: CGFloat { barHeight + rowsHeight }

    var onKey: ((Key) -> Void)?
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
    private var fnFill: UIColor { dark ? UIColor(white: 0.227, alpha: 1) : UIColor(red: 0.678, green: 0.702, blue: 0.737, alpha: 1) }
    private var ink: UIColor { dark ? .white : .black }
    private var muted: UIColor { dark ? UIColor(white: 0.82, alpha: 1) : UIColor(red: 0.235, green: 0.235, blue: 0.263, alpha: 1) }
    private var keyShadow: UIColor { dark ? UIColor(white: 0.08, alpha: 1) : UIColor(red: 0.537, green: 0.541, blue: 0.553, alpha: 1) }
    private var blue: UIColor { UIColor(red: 0, green: 0.478, blue: 1, alpha: 1) }
    private var blueShadow: UIColor { UIColor(red: 0.04, green: 0.37, blue: 0.75, alpha: 1) }

    override func traitCollectionDidChange(_ previous: UITraitCollection?) {
        super.traitCollectionDidChange(previous)
        if previous?.userInterfaceStyle != traitCollection.userInterfaceStyle { reload() }
    }

    // MARK: build

    private func build() {
        backgroundColor = bed
        bar.translatesAutoresizingMaskIntoConstraints = false
        addSubview(bar)
        addSubview(keysView)
        NSLayoutConstraint.activate([
            bar.topAnchor.constraint(equalTo: topAnchor, constant: 6),
            bar.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            bar.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            bar.heightAnchor.constraint(equalToConstant: 48),
        ])
        buildBar()
        buildRows()
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
            pill.topAnchor.constraint(equalTo: bar.topAnchor, constant: 4),
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
        r.append((key("space", .space), 0))
        r.append((key("sf:return", .ret, fn: true), 2.73))
        return r
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        keysView.frame = CGRect(x: 0, y: Self.barHeight + 6, width: bounds.width, height: Self.rowsHeight - 6)
        emojiView?.frame = bounds
        sheet?.frame = bounds
        let W = bounds.width, gap: CGFloat = 6, margin: CGFloat = 3
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
        backgroundColor = bed
        bar.backgroundColor = keyFill
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
        if sheetOpen { setSheet(true) }
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
            status.text = "Translation appears here as you type"
            status.font = .systemFont(ofSize: 12)
            status.textColor = dark ? UIColor(white: 0.55, alpha: 1) : UIColor(white: 0.62, alpha: 1)
            accent.backgroundColor = dark ? UIColor(white: 0.45, alpha: 1) : UIColor(white: 0.75, alpha: 1)
            actionBtn.isHidden = true
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
    }

    /// Keeps the previous preview visible while a newer one is in flight.
    func setTranslating() {
        let keep = state
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
        bar.isHidden = emoji
        keysView.isHidden = emoji
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
    }

    func showGlobe(_ on: Bool) { if on != needsGlobe { needsGlobe = on; buildRows() } }

    // MARK: language sheet

    func setSheet(_ open: Bool, side: LanguageSheet.Side = .to) {
        sheetOpen = open
        sheet?.removeFromSuperview(); sheet = nil
        keysView.isHidden = open
        bar.isHidden = open
        guard open else { return }
        let s = LanguageSheet(dark: dark, side: side) { [weak self] k in self?.onKey?(k) }
        addSubview(s)
        sheet = s
        setNeedsLayout()
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
        layer.cornerRadius = 7
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

/// Language + tone picker that takes the place of the key rows. `from` picks the typing
/// language (and key layout), `to` the translation target.
final class LanguageSheet: UIView {
    enum Side { case from, to }

    init(dark: Bool, side: Side, send: @escaping (KeyboardView.Key) -> Void) {
        super.init(frame: .zero)
        backgroundColor = dark ? UIColor(white: 0.17, alpha: 1) : UIColor(red: 0.82, green: 0.827, blue: 0.851, alpha: 1)
        let ink: UIColor = dark ? .white : .black
        let card: UIColor = dark ? UIColor(white: 0.353, alpha: 1) : .white
        let blue = UIColor(red: 0, green: 0.478, blue: 1, alpha: 1)
        let current = side == .to ? Prefs.targetCode : Prefs.sourceCode

        // Two selectors; the active one is highlighted and the list below fills it.
        func selector(_ label: String, _ code: String, active: Bool, _ k: KeyboardView.Key) -> UIButton {
            var c = UIButton.Configuration.filled()
            c.cornerStyle = .medium
            c.baseBackgroundColor = active ? blue : card
            c.contentInsets = .init(top: 6, leading: 12, bottom: 6, trailing: 10)
            let a = NSMutableAttributedString(string: label + "\n", attributes: [.font: UIFont.systemFont(ofSize: 11, weight: .medium), .foregroundColor: active ? UIColor(white: 1, alpha: 0.8) : UIColor.systemGray])
            a.append(NSAttributedString(string: Languages.by(code: code).name, attributes: [.font: UIFont.systemFont(ofSize: 15, weight: .semibold), .foregroundColor: active ? UIColor.white : ink]))
            c.attributedTitle = AttributedString(a)
            c.titleAlignment = .leading
            c.image = UIImage(systemName: "chevron.up.chevron.down", withConfiguration: UIImage.SymbolConfiguration(pointSize: 11, weight: .bold))
            c.imagePlacement = .trailing
            c.imagePadding = 6
            c.baseForegroundColor = active ? .white : .systemGray
            let b = UIButton(configuration: c)
            b.contentHorizontalAlignment = .fill
            b.addAction(UIAction { _ in send(k) }, for: .touchUpInside)
            return b
        }
        let fromSel = selector("I TYPE IN", Prefs.sourceCode, active: side == .from, .pickSource(Prefs.sourceCode))
        let toSel = selector("THEY RECEIVE IN", Prefs.targetCode, active: side == .to, .pickLanguage(Prefs.targetCode))
        let done = UIButton(type: .system)
        done.setTitle("Done", for: .normal)
        done.titleLabel?.font = .systemFont(ofSize: 17, weight: .semibold)
        done.addAction(UIAction { _ in send(.sheet(false)) }, for: .touchUpInside)
        done.setContentHuggingPriority(.required, for: .horizontal)
        let head = UIStackView(arrangedSubviews: [fromSel, toSel, done]); head.axis = .horizontal; head.spacing = 8
        head.distribution = .fill
        fromSel.widthAnchor.constraint(equalTo: toSel.widthAnchor).isActive = true

        // Recently used first, then the rest alphabetically; the current one is checked.
        let recent = (side == .to ? Prefs.recents : [Prefs.sourceCode]).map(Languages.by)
        let rest = Languages.all.filter { l in !recent.contains { $0.code == l.code } }.sorted { $0.name < $1.name }
        let list = UIStackView(); list.axis = .vertical
        list.backgroundColor = card; list.layer.cornerRadius = 10; list.clipsToBounds = true
        for (i, l) in (recent + rest).enumerated() {
            let row = UIButton(type: .system)
            var c = UIButton.Configuration.plain()
            c.contentInsets = .init(top: 0, leading: 14, bottom: 0, trailing: 14)
            c.titleAlignment = .leading
            let a = NSMutableAttributedString(string: l.code.padding(toLength: 4, withPad: " ", startingAt: 0), attributes: [.font: UIFont.monospacedSystemFont(ofSize: 12, weight: .bold), .foregroundColor: dark ? UIColor(white: 0.75, alpha: 1) : UIColor.darkGray])
            a.append(NSAttributedString(string: l.name + "  ", attributes: [.font: UIFont.systemFont(ofSize: 17), .foregroundColor: ink]))
            a.append(NSAttributedString(string: l.native, attributes: [.font: UIFont.systemFont(ofSize: 14), .foregroundColor: UIColor.systemGray]))
            c.attributedTitle = AttributedString(a)
            if l.code == current {
                c.image = UIImage(systemName: "checkmark", withConfiguration: UIImage.SymbolConfiguration(pointSize: 14, weight: .bold))
                c.imagePlacement = .trailing; c.baseForegroundColor = blue
            }
            row.configuration = c
            row.contentHorizontalAlignment = l.code == current ? .fill : .leading
            row.heightAnchor.constraint(equalToConstant: 44).isActive = true
            row.addAction(UIAction { _ in send(side == .to ? .pickLanguage(l.code) : .pickSource(l.code)) }, for: .touchUpInside)
            if i > 0 {
                let sep = UIView(); sep.backgroundColor = dark ? UIColor(white: 0.45, alpha: 1) : UIColor(white: 0.9, alpha: 1)
                sep.heightAnchor.constraint(equalToConstant: 0.5).isActive = true
                list.addArrangedSubview(sep)
            }
            list.addArrangedSubview(row)
        }
        let scroll = UIScrollView()
        scroll.disableEdgeBlur()
        scroll.addSubview(list)
        list.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            list.topAnchor.constraint(equalTo: scroll.contentLayoutGuide.topAnchor),
            list.bottomAnchor.constraint(equalTo: scroll.contentLayoutGuide.bottomAnchor),
            list.leadingAnchor.constraint(equalTo: scroll.contentLayoutGuide.leadingAnchor),
            list.trailingAnchor.constraint(equalTo: scroll.contentLayoutGuide.trailingAnchor),
            list.widthAnchor.constraint(equalTo: scroll.frameLayoutGuide.widthAnchor),
        ])

        let v = UIStackView(arrangedSubviews: [head, scroll])
        v.axis = .vertical; v.spacing = 8
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
