# Kronos Translate keyboard (iOS)

System-style keyboard (QWERTY / AZERTY / QWERTZ by typing language, long-press accents, full
system emoji set) with a reply-style reference strip above the keys. The field always stays in
your language; the strip shows the translation live (debounced 0.5 s) and **Use** swaps it in
before you send (Undo restores). Tap the strip to pick the typing and target languages.

Translation runs on the iPhone: the keyboard extension links the Kronos runtime (`libkronos_2t.a`)
and bundles Hy-MT2-1.8B (`Keyboard/Engine/Hy-MT2-1.8B`, 881 MB NF4). Weights are mapped read-only
from the bundle: clean file-backed pages do not count against the extension's memory cap, which is
only ~77 MB on an iPhone 12 Pro Max (measured: 8 MB before load, 28 MB loaded, ~60 MB while
translating). Never read the weight file through the page cache to "warm" it: that counts and gets
the keyboard killed. Your last 5 sent messages are context. No network, no Full Access. Without a
bundled model it falls back to `kronos_server`.

Design: https://www.figma.com/design/USzClG0m8LjwRQwbfufETU

## Build

```bash
brew install xcodegen
cargo build --release --lib --target aarch64-apple-ios --target aarch64-apple-ios-sim
M=models/Hy-MT2-1.8B; D=ios/KronosTranslate/Keyboard/Engine/Hy-MT2-1.8B; mkdir -p $D
for f in kronos-nf4g128.k2tb config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja generation_config.json; do ln -f $M/$f $D/$f; done
cd ios/KronosTranslate && xcodegen generate
xcodebuild -project KronosTranslate.xcodeproj -scheme KronosTranslate \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath build \
  CODE_SIGN_IDENTITY=- CODE_SIGNING_REQUIRED=NO build
xcrun simctl install booted build/Build/Products/Debug-iphonesimulator/KronosTranslate.app
```

## Run

1. Settings › General › Keyboard › Keyboards › Add New Keyboard › Translate.
2. Hold the globe key in any app and pick "Kronos — Translate".

Fallback without a bundled model: run `kronos_server <model> --mode nf4g128 --metal --host 0.0.0.0
--port 18085` on the Mac, enter `http://<mac-lan-ip>:18085` in the app, tap Test, and allow Full Access.
The keyboard container's accessibility value (`kronos.board`) carries engine diagnostics (launch
count and what the previous instance was doing when it died, load/decode times, jobs, footprint and
headroom); typing `zzdiag` + space writes them into the field. `./test-device.sh` builds, signs and
runs `LocalTranslateTests` on the phone, re-enabling the keyboard in Settings if a reinstall dropped
it, and prints the translation times and diagnostics.

Layout: `App/` SwiftUI settings, `Keyboard/` the extension (UIKit), `Shared/` prefs (app group) and
the HTTP client. On Face ID iPhones iOS draws the globe and dictation keys itself, so the keyboard
only adds one when `needsInputModeSwitchKey` is true.
