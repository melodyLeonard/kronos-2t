import SwiftUI

struct SettingsView: View {
    @State private var server = Prefs.serverURL
    @State private var source = Prefs.sourceCode
    @State private var target = Prefs.targetCode
    @State private var contextOn = Prefs.contextOn
    @State private var status: Status = .idle
    @State private var sample = ""

    enum Status { case idle, testing, ok(TimeInterval), failed(String) }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack(spacing: 14) {
                        Image("Logo").resizable().frame(width: 64, height: 64).clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Kronos Translate").font(.title3.bold())
                            Text("Type in your language, send in theirs.").foregroundStyle(.secondary).font(.subheadline)
                        }
                    }
                    .listRowBackground(Color.clear)
                    .listRowInsets(.init(top: 0, leading: 4, bottom: 8, trailing: 0))
                }
                Section("Try it") {
                    TextField("Switch to the Kronos keyboard and type here", text: $sample, axis: .vertical).lineLimit(3...6)
                }
                Section("Set up") {
                    step(1, "Add the keyboard", "Settings › General › Keyboard › Keyboards › Add New Keyboard › Translate")
                    step(2, "Start typing", "Translation runs on this iPhone with the built-in model. No internet, no Full Access needed.")
                    Button {
                        if let u = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(u) }
                    } label: { Label("Open Settings", systemImage: "gear") }
                }
                Section("Engine") {
                    HStack {
                        Image(systemName: "iphone").settingsIcon(.green)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(Self.bundledModel == nil ? "No on-device model" : "On this iPhone · Hy-MT2 1.8B")
                            Text(Self.bundledModel.map { "\($0) · works offline" } ?? "Falls back to the Mac server below").font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                    HStack {
                        Image(systemName: "desktopcomputer").settingsIcon(.blue)
                        TextField("http://192.168.1.70:18085", text: $server)
                            .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                            .onChange(of: server) { Prefs.serverURL = server }
                    }
                    HStack {
                        statusRow
                        Spacer()
                        Button("Test") { test() }.disabled({ if case .testing = status { true } else { false } }())
                    }
                }
                Section {
                    Picker(selection: $source) {
                        ForEach(Languages.all) { Text("\($0.name) · \($0.native)").tag($0.code) }
                    } label: { Label("I type in", systemImage: "keyboard").settingsLabel(.gray) }
                    .onChange(of: source) { Prefs.sourceCode = source }
                    Picker(selection: $target) {
                        ForEach(Languages.all) { Text("\($0.name) · \($0.native)").tag($0.code) }
                    } label: { Label("Translate to", systemImage: "globe").settingsLabel(.orange) }
                    .onChange(of: target) { Prefs.targetCode = target; Prefs.touchRecent(target) }
                    Toggle(isOn: $contextOn) { Label("Use chat context", systemImage: "bubble.left.and.bubble.right").settingsLabel(.green) }
                        .onChange(of: contextOn) { Prefs.contextOn = contextOn }
                } header: {
                    Text("Defaults")
                } footer: {
                    Text("The typing language also picks the key layout (AZERTY for French, QWERTZ for German). Context keeps the last \(Prefs.contextMax) messages you sent, on this iPhone only, so names, pronouns and slang translate the way you meant them.")
                }
            }
            .navigationTitle("Translate")
        }
    }

    @ViewBuilder private var statusRow: some View {
        switch status {
        case .idle: Label("Not tested", systemImage: "circle.dashed").foregroundStyle(.secondary)
        case .testing: HStack { ProgressView(); Text("Testing…").foregroundStyle(.secondary) }
        case .ok(let t): Label(String(format: "Connected · %.1f s round trip", t), systemImage: "circle.fill").foregroundStyle(.green)
        case .failed(let m): Label(m, systemImage: "exclamationmark.circle").foregroundStyle(.red)
        }
    }

    private func step(_ n: Int, _ title: String, _ sub: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text("\(n)").font(.caption.bold()).foregroundStyle(.white).frame(width: 24, height: 24).background(Color.accentColor, in: Circle())
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                Text(sub).font(.footnote).foregroundStyle(.secondary)
            }
        }
    }

    /// Size of the model inside the keyboard extension, or nil when it is not bundled.
    private static var bundledModel: String? {
        let ext = Bundle.main.bundleURL.appendingPathComponent("PlugIns/KronosKeyboard.appex/Hy-MT2-1.8B/kronos-nf4g128.k2tb")
        guard let size = (try? ext.resourceValues(forKeys: [.fileSizeKey]))?.fileSize else { return nil }
        return ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)
    }

    private func test() {
        status = .testing
        Task {
            switch await TranslateClient.ping() {
            case .success(let t): status = .ok(t)
            case .failure(let e): status = .failed(e.localizedDescription)
            }
        }
    }
}

private extension View {
    func settingsIcon(_ color: Color) -> some View {
        foregroundStyle(.white).font(.system(size: 14, weight: .semibold)).frame(width: 29, height: 29)
            .background(color, in: RoundedRectangle(cornerRadius: 7))
    }
}

private extension Label<Text, Image> {
    func settingsLabel(_ color: Color) -> some View {
        labelStyle(IconTileLabelStyle(color: color))
    }
}

private struct IconTileLabelStyle: LabelStyle {
    let color: Color
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 12) {
            configuration.icon.settingsIcon(color)
            configuration.title
        }
    }
}
