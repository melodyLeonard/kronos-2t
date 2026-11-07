#!/bin/sh
# Build unsigned, sign app + keyboard with the team's wildcard dev profile, install on the phone.
# Avoids Xcode's automatic signing (needs the account holder to accept Apple's PLA).
set -e
cd "$(dirname "$0")"
DEVICE=${DEVICE:-00008101-001C31941169001E}
TEAM=CU42G7MK57
ID="Apple Development: Melody Ifeanyi (VHM6N346TX)"
PROF=$(for f in ~/Library/Developer/Xcode/UserData/Provisioning\ Profiles/*.mobileprovision; do
  security cms -D -i "$f" | grep -q "<string>iOS Team Provisioning Profile: \*</string>" && echo "$f" && break; done)
[ -n "$PROF" ] || { echo "no wildcard profile for $TEAM"; exit 1; }

xcodegen generate >/dev/null
xcodebuild -project KronosTranslate.xcodeproj -scheme KronosTranslate -destination 'generic/platform=iOS' \
  -configuration Debug -derivedDataPath build-device CODE_SIGNING_ALLOWED=NO ENABLE_DEBUG_DYLIB=NO build 2>&1 | grep -E "error:|BUILD"

OUT=build-device/Build/Products/Debug-iphoneos
APP=$OUT/KronosTranslate.app
EXT=$APP/PlugIns/KronosKeyboard.appex
ent() { cat > "$1" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>application-identifier</key><string>$TEAM.$2</string>
<key>com.apple.developer.team-identifier</key><string>$TEAM</string>
<key>get-task-allow</key><true/>
</dict></plist>
EOF
}
ent "$OUT/ent-app.plist" com.melodyleonard.KronosTranslate
ent "$OUT/ent-ext.plist" com.melodyleonard.KronosTranslate.keyboard
cp "$PROF" "$EXT/embedded.mobileprovision"; cp "$PROF" "$APP/embedded.mobileprovision"
# Any leftover nested binaries (debug dylibs, frameworks) must be signed before their bundle.
find "$APP" -name "*.dylib" -o -name "*.framework" | while read -r f; do codesign --force --sign "$ID" --timestamp=none "$f"; done
codesign --force --sign "$ID" --entitlements "$OUT/ent-ext.plist" --timestamp=none "$EXT"
codesign --force --sign "$ID" --entitlements "$OUT/ent-app.plist" --timestamp=none "$APP"
codesign --verify --deep --strict "$APP"
xcrun devicectl device install app --device "$DEVICE" "$APP" | grep -i installationURL
xcrun devicectl device process launch --device "$DEVICE" com.melodyleonard.KronosTranslate >/dev/null
echo installed
