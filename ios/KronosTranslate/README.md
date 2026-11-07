# Kronos Translate keyboard (iOS)

System-style keyboard with one extra bar: language pill · status · Translate. Type English, tap
Translate, the field is replaced with the translation (Undo restores it), then send as usual.
Translation runs on Kronos `POST /translate` with tone and your last 5 sent messages as context.

Design: https://www.figma.com/design/USzClG0m8LjwRQwbfufETU

## Build

```bash
brew install xcodegen
cd ios/KronosTranslate && xcodegen generate
xcodebuild -project KronosTranslate.xcodeproj -scheme KronosTranslate \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath build \
  CODE_SIGN_IDENTITY=- CODE_SIGNING_REQUIRED=NO build
xcrun simctl install booted build/Build/Products/Debug-iphonesimulator/KronosTranslate.app
```

## Run

1. On the Mac: `kronos_server <model> --mode nf4g128 --metal --host 0.0.0.0 --port 18085`.
2. In the app, enter `http://<mac-lan-ip>:18085` and tap Test.
3. Settings › General › Keyboard › Keyboards › Add New Keyboard › Translate, then Allow Full Access.
4. Hold the globe key in any app and pick "Kronos — Translate".

Layout: `App/` SwiftUI settings, `Keyboard/` the extension (UIKit), `Shared/` prefs (app group) and
the HTTP client. On Face ID iPhones iOS draws the globe and dictation keys itself, so the keyboard
only adds one when `needsInputModeSwitchKey` is true.
