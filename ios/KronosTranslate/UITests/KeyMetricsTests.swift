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
    }

    private func dump(_ app: XCUIApplication, _ tag: String) {
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
