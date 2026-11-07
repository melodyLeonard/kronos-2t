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
    private let debounceInterval: TimeInterval = 0.5

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
        let h = view.heightAnchor.constraint(equalToConstant: KeyboardView.totalHeight)
        h.priority = .init(999)
        h.isActive = true
        board.onKey = { [weak self] in self?.handle($0) }
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
    private func textChanged() {
        let text = fieldText()
        if let t = inserted, text.isEmpty {
            // The field emptied after an insert: the host app sent it. Keep it as context.
            if Prefs.contextOn { Prefs.remember(sent: t) }
            inserted = nil; original = nil
        }
        if text.isEmpty, board.shift == .off { board.setShift(.on) }
        if let t = inserted, text == t { return }          // translation is in the field; nothing to preview
        inserted = nil; original = nil
        scheduleTranslate(text)
    }

    // MARK: keys

    private func handle(_ key: KeyboardView.Key) {
        switch key {
        case .char(let s):
            textDocumentProxy.insertText(board.shift == .off ? s : s.uppercased())
            if board.shift == .on { board.setShift(.off) }
        case .space:
            textDocumentProxy.insertText(" ")
        case .backspace:
            textDocumentProxy.deleteBackward()
            if fieldText().isEmpty { board.setShift(.on) }
        case .ret:
            textDocumentProxy.insertText("\n")
        case .shift:
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
            board.reload(); board.setSheet(true, side: .to)
        case .pickSource(let code):
            if code == Prefs.targetCode { Prefs.targetCode = Prefs.sourceCode }
            Prefs.sourceCode = code; sheetSide = .from
            cache.removeAll(); previewFor = ""
            board.reload(); board.setSheet(true, side: .from)
        case .tone(let t):
            Prefs.tone = t; cache.removeAll(); previewFor = ""
            board.reload(); board.setSheet(true, side: sheetSide)
        case .sheet(let open):
            board.setSheet(open, side: sheetSide)
            if !open { scheduleTranslate(fieldText(), now: true) }
        }
        UIDevice.current.playInputClick()
        // The proxy's document context updates after the current run loop turn.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self] in self?.textChanged() }
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

    private func cacheKey(_ t: String) -> String { "\(Prefs.sourceCode)>\(Prefs.targetCode)|\(Prefs.tone)|\(t)" }

    private func translate(_ text: String) {
        guard hasFullAccess else { board.flash("Allow Full Access in Settings"); return }
        task?.cancel()
        let from = Prefs.sourceCode, code = Prefs.targetCode, tone = Prefs.tone
        let ctx = Prefs.contextOn ? Prefs.context : []
        task = Task { [weak self] in
            do {
                let out = try await TranslateClient.translate(text, from: from, to: code, tone: tone, context: ctx)
                guard !Task.isCancelled, let self else { return }
                self.task = nil
                self.cache[self.cacheKey(text)] = out
                // Only show it if the field still says the same thing.
                guard self.fieldText().trimmingCharacters(in: .whitespacesAndNewlines) == text else { return }
                self.previewFor = text
                self.board.setState(.preview(out))
            } catch {
                guard let self, !Task.isCancelled else { return }
                self.task = nil
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
