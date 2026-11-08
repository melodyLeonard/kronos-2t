import UIKit

/// Translate keyboard. The field always stays in the typing language; the translation is shown
/// live in the reference strip above the keys and swapped in with one tap before sending.
final class KeyboardViewController: UIInputViewController {
    private var board: KeyboardView!
    private var original: String?          // text before "Use", for Undo
    private var inserted: String?          // translation currently in the field
    private var task: Task<Void, Never>?
    private var debounce: Timer?
    private var previewFor = ""            // source text the current preview belongs to
    private var cache: [String: String] = [:]
    private var sheetSide: LanguageSheet.Side = .to
    private let debounceInterval: TimeInterval = 0.14
    private var heightConstraint: NSLayoutConstraint?
    private let checker = UITextChecker()
    private var lastSpace = Date.distantPast, lastShift = Date.distantPast

    override func viewDidLoad() {
        super.viewDidLoad()
        board = KeyboardView(needsGlobe: needsInputModeSwitchKey)
        board.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(board)
        NSLayoutConstraint.activate([
            board.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            board.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            board.topAnchor.constraint(equalTo: view.topAnchor),
            board.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
        let h = view.heightAnchor.constraint(equalToConstant: board.height)
        h.priority = .init(999)
        h.isActive = true
        heightConstraint = h
        board.onKey = { [weak self] in self?.handle($0) }
        board.onSuggestion = { [weak self] in self?.accept(suggestion: $0) }
        board.onCursorMove = { [weak self] in self?.textDocumentProxy.adjustTextPosition(byCharacterOffset: $0) }
        board.onHeightChange = { [weak self] in self?.heightConstraint?.constant = $0 }
        updateDiagnostics()
        if LocalEngine.isBundled {
            LocalEngine.shared.load { [weak self] state in
                guard let self else { return }
                if case .unavailable(let why) = state { self.board.flash(why) }
                self.updateDiagnostics()
                self.scheduleTranslate(self.fieldText(), now: true)
            }
        }
    }

    /// Load time, last decode and memory headroom, readable by the UI test through the strip label.
    private var lastDecode = "", lastError = ""
    private func updateDiagnostics() {
        board.diagnostics = String(format: "load %.0f ms · warm %.0f ms · %d thr", LocalEngine.shared.loadMillis, LocalEngine.shared.warmMillis, ProcessInfo.processInfo.activeProcessorCount) + (lastDecode.isEmpty ? "" : " · \(lastDecode)") + " · " + LocalEngine.memorySummary
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        board.showGlobe(needsInputModeSwitchKey)
        board.reload()
        board.setShift(fieldText().isEmpty ? .on : .off)
        textChanged()
    }

    override func textDidChange(_ textInput: UITextInput?) {
        textChanged()
    }

    /// Called after every edit: keeps shift/context bookkeeping and schedules the live preview.
    private func textChanged(immediate: Bool = false) {
        let text = fieldText()
        if let t = inserted, text.isEmpty {
            // The field emptied after an insert: the host app sent it. Keep it as context.
            if Prefs.contextOn { Prefs.remember(sent: t) }
            inserted = nil; original = nil
        }
        if text.isEmpty, board.shift == .off { board.setShift(.on) }
        refreshSuggestions()
        if let t = inserted, text == t { return }          // translation is in the field; nothing to preview
        inserted = nil; original = nil
        scheduleTranslate(text, now: immediate)
    }

    // MARK: suggestions / autocorrect (UITextChecker, the system's own dictionaries)

    private var checkerLanguage: String {
        let code = Prefs.sourceCode.lowercased()
        return code == "en" ? "en_US" : (code == "zt" ? "zh-Hant" : code)
    }

    /// The word being typed: everything after the last whitespace before the caret.
    private var currentWord: String {
        let before = textDocumentProxy.documentContextBeforeInput ?? ""
        if let r = before.rangeOfCharacter(from: .whitespacesAndNewlines, options: .backwards) { return String(before[r.upperBound...]) }
        return before
    }

    private let suggestQueue = DispatchQueue(label: "kronos.suggest", qos: .userInteractive)
    private var suggestGen = 0

    /// Dictionary lookups take tens of ms; they never run on the key-handling thread.
    private func refreshSuggestions() {
        suggestGen += 1
        let gen = suggestGen
        let word = currentWord
        let before = textDocumentProxy.documentContextBeforeInput ?? ""
        let lang = checkerLanguage, src = Prefs.sourceCode
        suggestQueue.async { [weak self] in
            guard let self else { return }
            let s = self.suggestions(word: word, before: before, lang: lang, source: src)
            DispatchQueue.main.async { if gen == self.suggestGen { self.board.setSuggestions(s) } }
        }
    }

    private func suggestions(word: String, before: String, lang checkerLanguage: String, source: String) -> [String] {
        if word.isEmpty {
            // Sentence start: the system's usual openers; otherwise nothing.
            let sentenceStart = before.trimmingCharacters(in: .whitespaces).isEmpty || before.trimmingCharacters(in: .whitespaces).last.map { ".!?".contains($0) } == true
            return sentenceStart && source == "EN" ? ["I", "The", "I'm"] : []
        }
        let ns = word as NSString
        let all = NSRange(location: 0, length: ns.length)
        let miss = checker.rangeOfMisspelledWord(in: word, range: all, startingAt: 0, wrap: false, language: checkerLanguage)
        var out: [String] = []
        if miss.location != NSNotFound {
            out += checker.guesses(forWordRange: all, in: word, language: checkerLanguage) ?? []
        } else {
            out += checker.completions(forPartialWordRange: all, in: word, language: checkerLanguage) ?? []
        }
        out = out.filter { $0.lowercased() != word.lowercased() }
        let capped = Array(out.prefix(2)).map { matchCase($0, to: word) }
        return ["\u{201C}\(word)\u{201D}"] + capped
    }

    private func matchCase(_ s: String, to word: String) -> String {
        guard let f = word.first, f.isUppercase else { return s }
        return word.count > 1 && word == word.uppercased() ? s.uppercased() : s.prefix(1).uppercased() + s.dropFirst()
    }

    private func accept(suggestion: String) {
        replaceCurrentWord(with: suggestion + " ")
    }

    /// Space after a misspelled word applies the first guess, as the system keyboard does.
    private func autocorrectIfNeeded() {
        let word = currentWord
        guard word.count > 1, Prefs.sourceCode != "EN" || !word.contains("'") else { return }
        let ns = word as NSString
        let all = NSRange(location: 0, length: ns.length)
        let miss = checker.rangeOfMisspelledWord(in: word, range: all, startingAt: 0, wrap: false, language: checkerLanguage)
        guard miss.location == 0, miss.length == ns.length,
              let g = checker.guesses(forWordRange: all, in: word, language: checkerLanguage)?.first,
              g.count <= word.count + 2 else { return }
        replaceCurrentWord(with: matchCase(g, to: word))
    }

    private func replaceCurrentWord(with s: String) {
        for _ in 0..<currentWord.count { textDocumentProxy.deleteBackward() }
        textDocumentProxy.insertText(s)
    }

    // MARK: keys

    private func handle(_ key: KeyboardView.Key) {
        switch key {
        case .char(let s):
            let cased = Layouts.layout(for: Prefs.sourceCode).cased
            textDocumentProxy.insertText(board.shift == .off || !cased ? s : s.uppercased())
            if board.shift == .on { board.setShift(.off) }
        case .space:
            #if DEBUG
            if currentWord == "zzdiag" {   // test hook: engine diagnostics into the field
                updateDiagnostics()
                let st: String = { switch LocalEngine.shared.state { case .idle: return "idle"; case .loading: return "loading"; case .ready: return "ready"; case .unavailable(let w): return "unavailable: \(w)" } }()
                replaceCurrentWord(with: "[\(st) · \(board.diagnostics) · err \(lastError)]")
                break
            }
            #endif
            let before = textDocumentProxy.documentContextBeforeInput ?? ""
            if Date().timeIntervalSince(lastSpace) < 0.35, before.hasSuffix(" "), before.dropLast().last?.isLetter == true {
                // Double-tap space: ". " and capitalise, like the system keyboard.
                textDocumentProxy.deleteBackward()
                textDocumentProxy.insertText(". ")
                board.setShift(.on)
                lastSpace = .distantPast
                break
            }
            lastSpace = Date()
            autocorrectIfNeeded()
            textDocumentProxy.insertText(" ")
            // Sentence start capitalises, like the system keyboard.
            let tail = (textDocumentProxy.documentContextBeforeInput ?? "").trimmingCharacters(in: .whitespaces)
            if let l = tail.last, ".!?".contains(l) { board.setShift(.on) }
        case .backspace:
            textDocumentProxy.deleteBackward()
            if fieldText().isEmpty { board.setShift(.on) }
        case .ret:
            textDocumentProxy.insertText("\n")
        case .shift:
            if Date().timeIntervalSince(lastShift) < 0.35 { board.setShift(.lock); lastShift = .distantPast; break }
            lastShift = Date()
            board.setShift(board.shift == .off ? .on : .off)
        case .capsLock:
            board.setShift(.lock)
        case .globe:
            advanceToNextInputMode()
        case .page(let p):
            board.setPage(p)
        case .emoji(let e):
            textDocumentProxy.insertText(e)
        case .action:
            action()
        case .pickLanguage(let code):
            // Same language on both sides makes no sense: swap instead.
            if code == Prefs.sourceCode { Prefs.sourceCode = Prefs.targetCode }
            Prefs.targetCode = code; Prefs.touchRecent(code); sheetSide = .to
            cache.removeAll(); previewFor = ""
            board.reload()
        case .pickSource(let code):
            if code == Prefs.targetCode { Prefs.targetCode = Prefs.sourceCode }
            Prefs.sourceCode = code; sheetSide = .from
            cache.removeAll(); previewFor = ""
            board.reload()
        case .sheet(let open):
            board.setSheet(open, side: sheetSide)
            if !open { scheduleTranslate(fieldText(), now: true) }
        }
        UIDevice.current.playInputClick()
        // Trigger immediately on word/clause boundaries (space, return, sentence punctuation).
        let isWordBoundary: Bool = {
            switch key {
            case .space, .ret: return true
            case .char(let s): return ".!?,;:".contains(s)
            default: return false
            }
        }()
        DispatchQueue.main.async { [weak self] in self?.textChanged(immediate: isWordBoundary) }
    }

    private func fieldText() -> String {
        (textDocumentProxy.documentContextBeforeInput ?? "") + (textDocumentProxy.selectedText ?? "") + (textDocumentProxy.documentContextAfterInput ?? "")
    }

    // MARK: live preview

    private func scheduleTranslate(_ text: String, now: Bool = false) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        debounce?.invalidate()
        if t.isEmpty { task?.cancel(); task = nil; previewFor = ""; board.setState(.idle); return }
        if t == previewFor { return }
        if let hit = cache[cacheKey(t)] { previewFor = t; board.setState(.preview(hit)); return }
        board.setTranslating()
        debounce = Timer.scheduledTimer(withTimeInterval: now ? 0 : debounceInterval, repeats: false) { [weak self] _ in
            self?.translate(t)
        }
    }

    private func cacheKey(_ t: String) -> String { "\(Prefs.sourceCode)>\(Prefs.targetCode)|\(t)" }

    private func translate(_ text: String) {
        guard hasFullAccess || LocalEngine.isBundled else { board.flash("Allow Full Access in Settings"); return }
        task?.cancel()
        let from = Prefs.sourceCode, code = Prefs.targetCode
        let ctx = Prefs.contextOn ? Prefs.context : []
        let t0 = Date()
        task = Task { [weak self] in
            do {
                let out = try await TranslateClient.translateStream(text, from: from, to: code, context: ctx) { [weak self] partial in
                    guard let self, !Task.isCancelled else { return }
                    guard self.fieldText().trimmingCharacters(in: .whitespacesAndNewlines) == text else { return }
                    self.board.setStreamingPreview(partial, active: true)
                }
                guard !Task.isCancelled, let self else { return }
                self.task = nil
                self.cache[self.cacheKey(text)] = out
                // Only show it if the field still says the same thing.
                guard self.fieldText().trimmingCharacters(in: .whitespacesAndNewlines) == text else { return }
                self.previewFor = text
                self.board.setStreamingPreview(out, active: false)
                self.lastDecode = String(format: "%.2f s", Date().timeIntervalSince(t0))
                self.updateDiagnostics()
            } catch {
                guard let self, !Task.isCancelled else { return }
                self.task = nil
                self.lastError = error.localizedDescription
                self.board.flash(error.localizedDescription)
            }
        }
    }

    // MARK: use / undo

    private func action() {
        if let o = original, inserted != nil {
            replaceField(with: o)
            inserted = nil; original = nil
            previewFor = ""
            scheduleTranslate(o, now: true)
            return
        }
        guard case .preview(let t) = board.state else { return }
        original = fieldText()
        replaceField(with: t)
        inserted = t
        board.setState(.inserted(original ?? ""))
    }

    /// Replaces the whole field: jump to the end, delete everything, insert.
    private func replaceField(with text: String) {
        let after = textDocumentProxy.documentContextAfterInput ?? ""
        textDocumentProxy.adjustTextPosition(byCharacterOffset: after.utf16.count)
        let total = fieldText().count
        for _ in 0..<total { textDocumentProxy.deleteBackward() }
        textDocumentProxy.insertText(text)
    }
}
