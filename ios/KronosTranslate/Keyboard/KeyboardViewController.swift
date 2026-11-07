import UIKit

/// Translate keyboard: a system-style QWERTY with one extra bar (language pill · status · action).
final class KeyboardViewController: UIInputViewController {
    private var board: KeyboardView!
    private var original: String?          // text before the last translation, for Undo
    private var translated: String?        // what we put in the field
    private var task: Task<Void, Never>?

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
        board.globeHandler = { [weak self] in self?.advanceToNextInputMode() }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        board.showGlobe(needsInputModeSwitchKey)
        board.reload()
        board.setShift(fieldText().isEmpty ? .on : .off)
        refreshState()
    }

    override func textDidChange(_ textInput: UITextInput?) {
        // The field emptied after a translation: the host app sent it. Keep it as context.
        if let t = translated, fieldText().isEmpty {
            if Prefs.contextOn { Prefs.remember(sent: t) }
            translated = nil; original = nil
        }
        if fieldText().isEmpty, board.shift == .off { board.setShift(.on) }
        refreshState()
    }

    // MARK: keys

    private func handle(_ key: KeyboardView.Key) {
        switch key {
        case .char(let s):
            textDocumentProxy.insertText(board.shift == .off ? s : s.uppercased())
            if board.shift == .on { board.setShift(.off) }
            if translated != nil { translated = nil; original = nil }
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
        case .page(let p):
            board.setPage(p)
        case .emoji(let e):
            textDocumentProxy.insertText(e)
        case .action:
            action()
        case .pickLanguage(let code):
            Prefs.targetCode = code; Prefs.touchRecent(code); board.reload(); refreshState()
        case .tone(let t):
            Prefs.tone = t; board.reload(); refreshState()
        case .sheet(let open):
            board.setSheet(open); refreshState()
        }
        UIDevice.current.playInputClick()
        // The proxy's document context updates after the current run loop turn.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self] in self?.refreshState() }
    }

    private func fieldText() -> String {
        (textDocumentProxy.documentContextBeforeInput ?? "") + (textDocumentProxy.selectedText ?? "") + (textDocumentProxy.documentContextAfterInput ?? "")
    }

    private func refreshState() {
        if task != nil { board.setState(.translating) }
        else if translated != nil { board.setState(.translated) }
        else if fieldText().trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { board.setState(.idle) }
        else { board.setState(.ready) }
    }

    // MARK: translate / undo / cancel

    private func action() {
        if task != nil { task?.cancel(); task = nil; board.setState(.ready); return }
        if let o = original, translated != nil { replaceField(with: o); translated = nil; original = nil; return }
        let text = fieldText()
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard hasFullAccess else { board.flash("Allow Full Access in Settings"); return }
        let code = Prefs.targetCode, tone = Prefs.tone
        let ctx = Prefs.contextOn ? Prefs.context : []
        board.setState(.translating)
        task = Task { [weak self] in
            do {
                let out = try await TranslateClient.translate(text, to: code, tone: tone, context: ctx)
                guard !Task.isCancelled, let self else { return }
                self.task = nil
                self.original = text
                self.replaceField(with: out)
                self.translated = out
                self.board.setState(.translated)
            } catch {
                guard let self, !Task.isCancelled else { return }
                self.task = nil
                self.board.setState(.ready)
                self.board.flash(error.localizedDescription)
            }
        }
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
