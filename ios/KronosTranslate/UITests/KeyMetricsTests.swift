import XCTest

/// Dumps every key's frame (points) for whichever keyboard is up, switches with the globe key,
/// and dumps again, so the Kronos and system layouts can be compared exactly rather than by eye.
/// Screenshots go to the app's tmp dir (path printed) because `simctl io screenshot` omits keyboards.
final class KeyMetricsTests: XCTestCase {
    func testDumpKeyFrames() {
        let app = XCUIApplication()
        app.launch()
        let field = app.textViews.firstMatch.exists ? app.textViews.firstMatch : app.textFields.firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        field.tap()
        sleep(2)
        dump(app, "a")
        let globe = app.buttons["Next keyboard"]
        XCTAssertTrue(globe.waitForExistence(timeout: 3))
        globe.tap()
        sleep(2)
        dump(app, "b")
        // Touch routing on our keyboard: each tap must land on the key under it, delete must delete.
        if app.keys["delete.left"].exists {
            app.keys["h"].tap(); app.keys["l"].tap(); app.keys["delete.left"].tap()
            sleep(1)
            XCTAssertEqual((field.value as? String) ?? "", "H")
            // Number and symbol pages: every key fully on screen, delete and return included.
            let screenW = XCUIScreen.main.screenshot().image.size.width
            app.keys["123"].tap(); sleep(1)
            for page in ["numbers", "symbols"] {
                for k in app.keys.allElementsBoundByIndex {
                    let f = k.frame
                    XCTAssertTrue(f.minX >= 0 && f.maxX <= screenW + 0.5, "\(page) key \(k.label) off screen: \(f)")
                }
                XCTAssertTrue(app.keys["delete.left"].exists && app.keys["return"].exists, "\(page): delete/return missing")
                if page == "numbers" { app.keys["#+="].tap(); sleep(1) }
            }
            app.keys["ABC"].tap()
        }
    }

    func dump(_ app: XCUIApplication, _ tag: String) {
        for line in app.debugDescription.split(separator: "\n") {
            let s = line.trimmingCharacters(in: .whitespaces)
            if s.hasPrefix("Key,") || s.hasPrefix("Button,") || s.hasPrefix("Keyboard,") { print("TREE \(tag) \(s)") }
        }
        let shots: [(String, Data)] = [
            ("screen", XCUIScreen.main.screenshot().pngRepresentation),
            ("app", app.screenshot().pngRepresentation),
            ("springboard", XCUIApplication(bundleIdentifier: "com.apple.springboard").screenshot().pngRepresentation),
        ]
        for (kind, png) in shots {
            let path = NSTemporaryDirectory() + "kb-\(tag)-\(kind).png"
            try? png.write(to: URL(fileURLWithPath: path))
            print("PNG \(tag) \(kind) \(path)")
        }
    }
}

/// Types on the Kronos keyboard and waits for the on-device translation; prints the strip text and
/// the engine diagnostics (load time, decode time, memory headroom under the extension's cap),
/// which the keyboard exposes as the accessibility value of its `kronos.board` container.
final class LocalTranslateTests: XCTestCase {
    func testTypeAndTranslate() {
        let app = XCUIApplication()
        app.launch()
        let field = app.textViews.firstMatch.exists ? app.textViews.firstMatch : app.textFields.firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        field.tap()
        sleep(2)
        if !app.buttons["Next keyboard"].exists, !app.keys["delete.left"].exists { enableKeyboard(app); field.tap(); sleep(2) }
        // Cycle keyboards (system, emoji, ...) until ours is up.
        for _ in 0..<4 where !app.keys["delete.left"].exists {
            let globe = app.buttons["Next keyboard"]
            if globe.waitForExistence(timeout: 3) { globe.tap() } else { app.coordinate(withNormalizedOffset: CGVector(dx: 0.08, dy: 0.97)).tap() }
            print("TREE cycle globe=\(app.buttons["Next keyboard"].exists) kronos=\(app.keys["delete.left"].exists)")
            sleep(2)
        }
        if !app.keys["delete.left"].exists { KeyMetricsTests().dump(app, "notup") }
        XCTAssertTrue(app.keys["delete.left"].waitForExistence(timeout: 3), "Kronos keyboard not up")
        let boardEl = app.otherElements["kronos.board"]
        let status = app.staticTexts["kronos.status"]
        let diag = { boardEl.exists ? (boardEl.value as? String ?? "") : "missing" }
        print("DIAG start \(diag())")
        let t1 = Date()
        for k in ["h", "o", "w", "space", "a", "r", "e", "space", "y", "o", "u", "space", "t", "o", "d", "a", "y"] {
            if !app.keys[k].waitForExistence(timeout: 1) {
                print("DIAG typing: key '\(k)' missing after \(String(format: "%.1f", Date().timeIntervalSince(t1))) s; kb=\(app.keys["delete.left"].exists)")
                for line in app.debugDescription.split(separator: "\n") where line.contains("Key, ") || line.contains("kronos") { print("TREE miss \(line.trimmingCharacters(in: .whitespaces).prefix(160))") }
                for _ in 0..<4 where !app.keys["delete.left"].exists {
                    let globe = app.buttons["Next keyboard"]
                    if globe.waitForExistence(timeout: 3) { globe.tap() }
                    sleep(3)
                }
                print("DIAG typing relaunch kb=\(app.keys["delete.left"].exists) \(diag())")
                XCTAssertTrue(app.keys[k].waitForExistence(timeout: 2), "keyboard did not come back")
            }
            app.keys[k].tap()
        }
        func waitTranslation(_ tag: String, _ typed: String) {
            let t0 = Date()
            for _ in 0..<60 {
                sleep(2)
                if !app.keys["delete.left"].exists {
                    print("DIAG \(tag) keyboard gone after \(String(format: "%.1f", Date().timeIntervalSince(t0))) s")
                    for _ in 0..<4 where !app.keys["delete.left"].exists {
                        let globe = app.buttons["Next keyboard"]
                        if globe.waitForExistence(timeout: 3) { globe.tap() }
                        sleep(3)
                    }
                    print("DIAG \(tag) relaunch kb=\(app.keys["delete.left"].exists) \(diag())")
                    return
                }
                let s = status.exists ? status.label : "-"
                if !s.isEmpty, s != "-", s != "…", !s.contains(typed) {
                    print("TRANSLATION \(tag) \(String(format: "%.1f", Date().timeIntervalSince(t0))) s: \(s)")
                    return
                }
            }
            print("TRANSLATION \(tag) timed out")
        }
        waitTranslation("first", "how are you")
        for k in ["space", "z", "z", "d", "i", "a", "g", "space"] { app.keys[k].tap() }
        sleep(1)
        print("DIAG first \(field.value as? String ?? "") | \(diag())")
        for k in Array("and what are you doing this weekend with the kids") { app.keys[k == " " ? "space" : String(k)].tap() }
        app.keys["space"].tap()
        waitTranslation("second", "zzdiag")
        for k in ["z", "z", "d", "i", "a", "g", "space"] { app.keys[k].tap() }
        sleep(1)
        print("DIAG second \(field.value as? String ?? "") | \(diag())")
        XCTAssertTrue(status.exists, "no translation shown")
    }
}

extension LocalTranslateTests {
    /// Reinstalling the app drops the keyboard from the enabled list; turn it back on via Settings.
    func enableKeyboard(_ app: XCUIApplication) {
        app.buttons["Open Settings"].tap()
        let settings = XCUIApplication(bundleIdentifier: "com.apple.Preferences")
        XCTAssertTrue(settings.wait(for: .runningForeground, timeout: 10))
        sleep(2)
        let kb = settings.staticTexts["Keyboards"]
        if kb.waitForExistence(timeout: 5) { kb.tap(); sleep(2) } else { print("TREE settings \(settings.debugDescription.prefix(3000))") }
        let sw = settings.switches["Kronos"]
        print("SWITCH before \(sw.exists) \(sw.value as? String ?? "?")")
        if sw.exists, (sw.value as? String) == "0" { sw.tap(); sleep(2) }
        if settings.alerts.firstMatch.exists { print("SWITCH alert \(settings.alerts.firstMatch.label)"); settings.alerts.buttons.firstMatch.tap() }
        print("SWITCH after \(sw.value as? String ?? "?")")
        app.activate()
        sleep(2)
    }
}
