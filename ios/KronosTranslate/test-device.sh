#!/bin/sh
# Builds the UI tests unsigned, signs app + keyboard + test runner with the wildcard dev profile and
# runs the given tests on the phone. usage: ./test-device.sh [KronosTranslateUITests/LocalTranslateTests]
set -e
cd "$(dirname "$0")"
DEVICE=${DEVICE:-00008101-001C31941169001E}
TEAM=CU42G7MK57
ID="Apple Development: Melody Ifeanyi (VHM6N346TX)"
ONLY=${1:-KronosTranslateUITests/LocalTranslateTests}
PROF=$(for f in ~/Library/Developer/Xcode/UserData/Provisioning\ Profiles/*.mobileprovision; do
  security cms -D -i "$f" | grep -q "<string>iOS Team Provisioning Profile: \*</string>" && echo "$f" && break; done)
[ -n "$PROF" ] || { echo "no wildcard profile for $TEAM"; exit 1; }

xcodegen generate >/dev/null
xcodebuild -project KronosTranslate.xcodeproj -scheme KronosTranslate -destination 'generic/platform=iOS' \
  -configuration Debug -derivedDataPath build-device CODE_SIGNING_ALLOWED=NO ENABLE_DEBUG_DYLIB=NO build-for-testing 2>&1 | grep -E "error:|BUILD"

OUT=build-device/Build/Products/Debug-iphoneos
APP=$OUT/KronosTranslate.app
EXT=$APP/PlugIns/KronosKeyboard.appex
RUN=$OUT/KronosTranslateUITests-Runner.app
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
ent "$OUT/ent-test.plist" com.melodyleonard.KronosTranslateUITests
ent "$OUT/ent-run.plist" com.melodyleonard.KronosTranslateUITests.xctrunner
for b in "$APP" "$RUN"; do cp "$PROF" "$b/embedded.mobileprovision"; done
cp "$PROF" "$EXT/embedded.mobileprovision"
find "$APP" "$RUN" -name "*.dylib" -o -name "*.framework" | while read -r f; do codesign --force --sign "$ID" --timestamp=none "$f"; done
codesign --force --sign "$ID" --entitlements "$OUT/ent-ext.plist" --timestamp=none "$EXT"
codesign --force --sign "$ID" --entitlements "$OUT/ent-app.plist" --timestamp=none "$APP"
codesign --force --sign "$ID" --entitlements "$OUT/ent-test.plist" --timestamp=none "$RUN/PlugIns/KronosTranslateUITests.xctest"
codesign --force --sign "$ID" --entitlements "$OUT/ent-run.plist" --timestamp=none "$RUN"
XCTR=$(ls build-device/Build/Products/*.xctestrun | head -1)
xcodebuild test-without-building -xctestrun "$XCTR" -destination "id=$DEVICE" -only-testing:"$ONLY" 2>&1 \
  | grep -E "TREE|PNG|TRANSLATION|DIAG|error:|crashed|Executed|passed|failed"
