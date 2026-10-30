//! Native zero-crate macOS top menu bar launcher and global shortcut Quick Prompt HUD for Kronos.
//!
//! Usage: `kronos_menubar [--url http://127.0.0.1:18080]`
//! Global shortcuts: `Option+Space` (`⌥Space`) and `Cmd+Shift+K` (`⌘⇧K`).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("kronos_menubar is only supported on macOS");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

#[cfg(target_os = "macos")]
mod macos {
    use kronos_2t::agent::mime_type_for_path;
    use kronos_2t::json::quote;
    use std::cell::RefCell;
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::fs;
    use std::path::Path;
    use std::process::Command;
    use std::ptr;

    type Id = *mut c_void;
    type Sel = *mut c_void;

    macro_rules! embed_info_plist {
        ($bytes:expr) => {
            const _: () = {
                #[used]
                #[link_section = "__TEXT,__info_plist"]
                static INFO_PLIST: [u8; $bytes.len()] = {
                    let mut out = [0u8; $bytes.len()];
                    let mut i = 0;
                    while i < $bytes.len() {
                        out[i] = $bytes[i];
                        i += 1;
                    }
                    out
                };
            };
        };
    }

    embed_info_plist!(
        br#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.melodyleonard.kronos.menubar</string>
    <key>CFBundleName</key>
    <string>Kronos</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>Kronos uses the microphone for local on-device voice commands.</string>
</dict>
</plist>
"#
    );

    #[repr(C)]
    struct ObjCBlockLiteral {
        isa: *const c_void,
        flags: i32,
        reserved: i32,
        invoke: Option<unsafe extern "C" fn(*mut ObjCBlockLiteral, isize)>,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct NSPoint {
        x: f64,
        y: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct NSSize {
        width: f64,
        height: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct NSRect {
        origin: NSPoint,
        size: NSSize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct EventTypeSpec {
        event_class: u32,
        event_kind: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct EventHotKeyID {
        signature: u32,
        id: u32,
    }

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_allocateClassPair(superclass: Id, name: *const c_char, extra_bytes: usize) -> Id;
        fn objc_registerClassPair(cls: Id);
        fn class_addMethod(cls: Id, name: Sel, imp: *const c_void, types: *const c_char) -> bool;
        fn objc_msgSend();
        fn objc_autoreleasePoolPush() -> *mut c_void;
        fn objc_autoreleasePoolPop(pool: *mut c_void);
    }

    #[link(name = "Foundation", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "WebKit", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "QuartzCore", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGMainDisplayID() -> u32;
        fn CGDisplayBounds(display: u32) -> NSRect;
        fn CGColorSpaceCreateDeviceRGB() -> *mut c_void;
        fn CGColorSpaceRelease(cs: *mut c_void);
        fn CGBitmapContextCreate(
            data: *mut c_void,
            width: usize,
            height: usize,
            bits_per_component: usize,
            bytes_per_row: usize,
            space: *mut c_void,
            bitmap_info: u32,
        ) -> *mut c_void;
        fn CGBitmapContextCreateImage(ctx: *mut c_void) -> *mut c_void;
        fn CGContextRelease(ctx: *mut c_void);
        fn CGImageRelease(img: *mut c_void);
    }

    #[link(name = "Carbon", kind = "framework")]
    unsafe extern "C" {
        fn GetApplicationEventTarget() -> *mut c_void;
        fn InstallEventHandler(
            in_target: *mut c_void,
            in_handler: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i32,
            in_num_types: u32,
            in_list: *const EventTypeSpec,
            in_user_data: *mut c_void,
            out_ref: *mut *mut c_void,
        ) -> i32;
        fn RegisterEventHotKey(
            in_hot_key_code: u32,
            in_hot_key_modifiers: u32,
            in_hot_key_id: EventHotKeyID,
            in_target: *mut c_void,
            in_options: u32,
            out_ref: *mut *mut c_void,
        ) -> i32;
    }

    macro_rules! msg {
        ($obj:expr, $sel:expr; $ret:ty) => {{
            let fn_ptr: unsafe extern "C" fn(Id, Sel) -> $ret =
                std::mem::transmute(objc_msgSend as *const ());
            fn_ptr($obj, sel($sel))
        }};
        ($obj:expr, $sel:expr, $($arg:expr => $aty:ty),+ ; $ret:ty) => {{
            let fn_ptr: unsafe extern "C" fn(Id, Sel, $($aty),+) -> $ret =
                std::mem::transmute(objc_msgSend as *const ());
            fn_ptr($obj, sel($sel), $($arg),+)
        }};
    }

    const PANEL_WIDTH: f64 = 680.0;
    const PANEL_INIT_HEIGHT: f64 = 142.0;

    struct BarState {
        base_url: String,
        panel: Id,
        webview: Id,
        status_button: Id,
        status_menu: Id,
        panel_x: f64,
        panel_top_y: f64,
        panel_height: f64,
        has_custom_pos: bool,
        dragging: bool,
        suppress_auto_hide: bool,
        drag_mouse_start: (f64, f64),
        drag_panel_start: (f64, f64),
        visible: bool,
    }

    thread_local! {
        static STATE: RefCell<Option<BarState>> = const { RefCell::new(None) };
    }

    fn class(name: &str) -> Id {
        let c = CString::new(name).unwrap();
        unsafe { objc_getClass(c.as_ptr()) }
    }

    fn sel(name: &str) -> Sel {
        let c = CString::new(name).unwrap();
        unsafe { sel_registerName(c.as_ptr()) }
    }

    unsafe fn ns_string(s: &str) -> Id {
        let c = CString::new(s).unwrap_or_else(|_| CString::new("").unwrap());
        msg!(class("NSString"), "stringWithUTF8String:", c.as_ptr() => *const c_char; Id)
    }

    unsafe fn ns_string_to_rust(ns: Id) -> Option<String> {
        if ns.is_null() {
            return None;
        }
        let ptr: *const c_char = msg!(ns, "UTF8String"; *const c_char);
        if ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }

    fn encode_base64(bytes: &[u8]) -> String {
        const LUT: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
        for ch in bytes.chunks(3) {
            let b0 = ch[0];
            let b1 = if ch.len() > 1 { ch[1] } else { 0 };
            let b2 = if ch.len() > 2 { ch[2] } else { 0 };
            s.push(LUT[(b0 >> 2) as usize] as char);
            s.push(LUT[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
            s.push(if ch.len() > 1 {
                LUT[(((b1 & 0x0F) << 2) | (b2 >> 6)) as usize] as char
            } else {
                '='
            });
            s.push(if ch.len() > 2 {
                LUT[(b2 & 0x3F) as usize] as char
            } else {
                '='
            });
        }
        s
    }

    /// Renders a 36x36 (2x Retina for 18x18 pt) anti-aliased monochrome Kronecker starburst alpha mask.
    pub(super) fn render_template_icon_rgba(side: usize) -> Vec<u8> {
        let mut buf = vec![0u8; side * side * 4];
        let center = (side as f64 - 1.0) * 0.5;
        let outer_r = side as f64 * 0.41;
        let inner_hole = side as f64 * 0.085;
        let stroke_half = side as f64 * 0.048;
        let dirs: [(f64, f64); 4] = [
            (1.0, 0.0),
            (0.0, 1.0),
            (std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2),
            (std::f64::consts::FRAC_1_SQRT_2, -std::f64::consts::FRAC_1_SQRT_2),
        ];
        for y in 0..side {
            for x in 0..side {
                let dx = x as f64 - center;
                let dy = y as f64 - center;
                let r = (dx * dx + dy * dy).sqrt();
                let mut min_dist = f64::MAX;
                for (ux, uy) in dirs {
                    let proj = (dx * ux + dy * uy).clamp(-outer_r, outer_r);
                    let px = ux * proj;
                    let py = uy * proj;
                    let d = ((dx - px).powi(2) + (dy - py).powi(2)).sqrt();
                    if d < min_dist {
                        min_dist = d;
                    }
                }
                let ray_cov = (stroke_half + 0.65 - min_dist).clamp(0.0, 1.0);
                let hole_cov = (r - inner_hole + 0.5).clamp(0.0, 1.0);
                let alpha = (ray_cov * hole_cov * 255.0).round() as u8;
                let idx = (y * side + x) * 4;
                buf[idx + 3] = alpha;
            }
        }
        buf
    }

    unsafe fn build_status_template_image() -> Id {
        let side = 36usize;
        let mut pixels = render_template_icon_rgba(side);
        let cs = CGColorSpaceCreateDeviceRGB();
        if cs.is_null() {
            return ptr::null_mut();
        }
        // kCGImageAlphaPremultipliedLast = 1
        let ctx = CGBitmapContextCreate(
            pixels.as_mut_ptr() as *mut c_void,
            side,
            side,
            8,
            side * 4,
            cs,
            1,
        );
        CGColorSpaceRelease(cs);
        if ctx.is_null() {
            return ptr::null_mut();
        }
        let cg_img = CGBitmapContextCreateImage(ctx);
        CGContextRelease(ctx);
        if cg_img.is_null() {
            return ptr::null_mut();
        }
        let alloc: Id = msg!(class("NSImage"), "alloc"; Id);
        let ns_img: Id = msg!(
            alloc,
            "initWithCGImage:size:",
            cg_img => *mut c_void,
            NSSize { width: 18.0, height: 18.0 } => NSSize;
            Id
        );
        CGImageRelease(cg_img);
        if !ns_img.is_null() {
            let _: () = msg!(ns_img, "setTemplate:", true => bool; ());
        }
        ns_img
    }

    fn compute_default_anchor() -> (f64, f64) {
        let bounds = unsafe { CGDisplayBounds(CGMainDisplayID()) };
        let w = if bounds.size.width > 400.0 { bounds.size.width } else { 1440.0 };
        let h = if bounds.size.height > 300.0 { bounds.size.height } else { 900.0 };
        let x = bounds.origin.x + (w - PANEL_WIDTH) * 0.5;
        let top_y = bounds.origin.y + h * 0.78;
        (x.round(), top_y.round())
    }

    unsafe fn is_anchor_visible_on_any_screen(x: f64, top_y: f64) -> bool {
        let screens: Id = msg!(class("NSScreen"), "screens"; Id);
        if screens.is_null() {
            return true;
        }
        let count: usize = msg!(screens, "count"; usize);
        if count == 0 {
            return true;
        }
        let probe_left = x + 60.0;
        let probe_right = x + PANEL_WIDTH - 60.0;
        let probe_y = top_y - 14.0;
        for i in 0..count {
            let scr: Id = msg!(screens, "objectAtIndex:", i => usize; Id);
            if !scr.is_null() {
                let vf: NSRect = msg!(scr, "visibleFrame"; NSRect);
                let sx0 = vf.origin.x;
                let sx1 = vf.origin.x + vf.size.width;
                let sy0 = vf.origin.y;
                let sy1 = vf.origin.y + vf.size.height;
                if probe_right > sx0 && probe_left < sx1 && probe_y >= sy0 && probe_y <= sy1 {
                    return true;
                }
            }
        }
        false
    }

    unsafe fn clamp_top_below_menubar(x: f64, mut top_y: f64) -> f64 {
        let screens: Id = msg!(class("NSScreen"), "screens"; Id);
        if screens.is_null() {
            return top_y;
        }
        let count: usize = msg!(screens, "count"; usize);
        let mid_x = x + PANEL_WIDTH * 0.5;
        for i in 0..count {
            let scr: Id = msg!(screens, "objectAtIndex:", i => usize; Id);
            if !scr.is_null() {
                let full: NSRect = msg!(scr, "frame"; NSRect);
                if mid_x >= full.origin.x
                    && mid_x <= full.origin.x + full.size.width
                    && top_y >= full.origin.y
                    && top_y <= full.origin.y + full.size.height + 120.0
                {
                    let vf: NSRect = msg!(scr, "visibleFrame"; NSRect);
                    let max_top = vf.origin.y + vf.size.height;
                    if top_y > max_top {
                        top_y = max_top;
                    }
                    break;
                }
            }
        }
        top_y
    }

    unsafe fn reset_panel_position() {
        STATE.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let Some(st) = borrow.as_mut() else { return };
            let (ax, top_y) = compute_default_anchor();
            st.panel_x = ax;
            st.panel_top_y = top_y;
            st.has_custom_pos = false;
            st.dragging = false;
            let frame = NSRect {
                origin: NSPoint {
                    x: st.panel_x,
                    y: st.panel_top_y - st.panel_height,
                },
                size: NSSize {
                    width: PANEL_WIDTH,
                    height: st.panel_height,
                },
            };
            let _: () = msg!(st.panel, "setFrame:display:", frame => NSRect, true => bool; ());
        });
    }

    unsafe fn eval_js(webview: Id, script: &str) {
        if webview.is_null() {
            return;
        }
        let ns = ns_string(script);
        let _: () = msg!(
            webview,
            "evaluateJavaScript:completionHandler:",
            ns => Id,
            ptr::null_mut() => Id;
            ()
        );
    }

    unsafe fn inject_file_into_webview(webview: Id, path: &Path) -> bool {
        let Ok(bytes) = fs::read(path) else {
            return false;
        };
        if bytes.is_empty() || bytes.len() > (24 << 20) {
            return false;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("attachment");
        let mime_raw = mime_type_for_path(path);
        let mime = mime_raw.split(';').next().unwrap_or("application/octet-stream").trim();
        let data_url = format!("data:{mime};base64,{}", encode_base64(&bytes));
        let js = format!(
            "if (window.kronosReceiveNativeFile) window.kronosReceiveNativeFile({}, {}, {});",
            quote(name),
            quote(mime),
            quote(&data_url)
        );
        eval_js(webview, &js);
        true
    }

    unsafe fn show_panel(reset_chat: bool) {
        let setup = STATE.with(|cell| {
            let Ok(mut borrow) = cell.try_borrow_mut() else {
                return None;
            };
            let st = borrow.as_mut()?;
            if !st.has_custom_pos || !is_anchor_visible_on_any_screen(st.panel_x, st.panel_top_y) {
                let (ax, top_y) = compute_default_anchor();
                st.panel_x = ax;
                st.panel_top_y = top_y;
                st.has_custom_pos = false;
            }
            st.dragging = false;
            st.visible = true;
            let frame = NSRect {
                origin: NSPoint {
                    x: st.panel_x,
                    y: st.panel_top_y - st.panel_height,
                },
                size: NSSize {
                    width: PANEL_WIDTH,
                    height: st.panel_height,
                },
            };
            let quick_url = format!("{}/quick", st.base_url.trim_end_matches('/'));
            Some((st.panel, st.webview, frame, quick_url))
        });
        let Some((panel, webview, frame, quick_url)) = setup else {
            return;
        };
        let _: () = msg!(panel, "setFrame:display:", frame => NSRect, true => bool; ());

        let cur_url: Id = msg!(webview, "URL"; Id);
        let url_blank = if cur_url.is_null() {
            true
        } else {
            let abs: Id = msg!(cur_url, "absoluteString"; Id);
            ns_string_to_rust(abs)
                .map(|u| u.is_empty() || u == "about:blank")
                .unwrap_or(true)
        };
        let title_ns: Id = msg!(webview, "title"; Id);
        let title_ready = ns_string_to_rust(title_ns)
            .map(|t| t.contains("Kronos"))
            .unwrap_or(false);
        let is_loading: bool = msg!(webview, "isLoading"; bool);
        if !is_loading && (url_blank || !title_ready) {
            let ns_u = ns_string(&quick_url);
            let url_obj: Id = msg!(class("NSURL"), "URLWithString:", ns_u => Id; Id);
            if !url_obj.is_null() {
                let req: Id = msg!(
                    class("NSURLRequest"),
                    "requestWithURL:cachePolicy:timeoutInterval:",
                    url_obj => Id,
                    1usize => usize,
                    15.0f64 => f64;
                    Id
                );
                let _: Id = msg!(webview, "loadRequest:", req => Id; Id);
            }
        }

        let app: Id = msg!(class("NSApplication"), "sharedApplication"; Id);
        let _: () = msg!(app, "activateIgnoringOtherApps:", true => bool; ());
        let _: () = msg!(panel, "makeKeyAndOrderFront:", ptr::null_mut() => Id; ());
        let _: bool = msg!(panel, "makeFirstResponder:", webview => Id; bool);

        if reset_chat {
            eval_js(
                webview,
                "if (window.kronosNewChat) window.kronosNewChat();",
            );
        } else {
            eval_js(
                webview,
                "if (window.kronosFocusInput) window.kronosFocusInput();",
            );
        }
    }

    unsafe fn hide_panel() {
        let panel = STATE.with(|cell| {
            let Ok(mut borrow) = cell.try_borrow_mut() else {
                return ptr::null_mut();
            };
            let Some(st) = borrow.as_mut() else {
                return ptr::null_mut();
            };
            st.dragging = false;
            st.visible = false;
            st.panel
        });
        if !panel.is_null() {
            let _: () = msg!(panel, "orderOut:", ptr::null_mut() => Id; ());
        }
    }

    unsafe fn toggle_panel() {
        let currently_visible = STATE.with(|cell| {
            cell.try_borrow()
                .ok()
                .and_then(|b| {
                    b.as_ref().map(|st| {
                        let is_vis: bool = unsafe { msg!(st.panel, "isVisible"; bool) };
                        st.visible && is_vis
                    })
                })
                .unwrap_or(false)
        });
        if currently_visible {
            hide_panel();
        } else {
            show_panel(false);
        }
    }

    unsafe fn resize_panel(new_height: f64) {
        let clamped = new_height.clamp(124.0, 560.0);
        let target = STATE.with(|cell| {
            let Ok(mut borrow) = cell.try_borrow_mut() else {
                return None;
            };
            let st = borrow.as_mut()?;
            if (st.panel_height - clamped).abs() < 1.0 {
                return None;
            }
            st.panel_height = clamped;
            let frame = NSRect {
                origin: NSPoint {
                    x: st.panel_x,
                    y: st.panel_top_y - clamped,
                },
                size: NSSize {
                    width: PANEL_WIDTH,
                    height: clamped,
                },
            };
            Some((st.panel, frame))
        });
        if let Some((panel, frame)) = target {
            let _: () = msg!(panel, "setFrame:display:", frame => NSRect, true => bool; ());
            let _: () = msg!(panel, "invalidateShadow"; ());
        }
    }

    fn open_external_url(url: &str) {
        let _ = Command::new("/usr/bin/open").arg(url).spawn();
    }

    unsafe fn pick_files_with_open_panel() {
        let webview = STATE.with(|cell| {
            let mut borrow = cell.borrow_mut();
            if let Some(s) = borrow.as_mut() {
                s.suppress_auto_hide = true;
                s.webview
            } else {
                ptr::null_mut()
            }
        });
        if webview.is_null() {
            return;
        }
        let open_panel: Id = msg!(class("NSOpenPanel"), "openPanel"; Id);
        if open_panel.is_null() {
            STATE.with(|cell| {
                if let Some(s) = cell.borrow_mut().as_mut() {
                    s.suppress_auto_hide = false;
                }
            });
            return;
        }
        let _: () = msg!(open_panel, "setCanChooseFiles:", true => bool; ());
        let _: () = msg!(open_panel, "setCanChooseDirectories:", false => bool; ());
        let _: () = msg!(open_panel, "setAllowsMultipleSelection:", true => bool; ());
        let _: () = msg!(open_panel, "setLevel:", 4isize => isize; ());
        let resp: isize = msg!(open_panel, "runModal"; isize);
        if resp == 1 {
            let urls: Id = msg!(open_panel, "URLs"; Id);
            if !urls.is_null() {
                let count: usize = msg!(urls, "count"; usize);
                for i in 0..count {
                    let u: Id = msg!(urls, "objectAtIndex:", i => usize; Id);
                    if !u.is_null() {
                        let p_ns: Id = msg!(u, "path"; Id);
                        if let Some(p_str) = ns_string_to_rust(p_ns) {
                            let _ = inject_file_into_webview(webview, Path::new(&p_str));
                        }
                    }
                }
            }
        }
        STATE.with(|cell| {
            if let Some(s) = cell.borrow_mut().as_mut() {
                s.suppress_auto_hide = false;
            }
        });
        show_panel(false);
    }

    unsafe extern "C" fn on_window_resign_key(_this: Id, _cmd: Sel, _notif: Id) {
        let should_hide = STATE.with(|cell| {
            cell.try_borrow()
                .ok()
                .and_then(|b| {
                    b.as_ref()
                        .map(|st| st.visible && !st.suppress_auto_hide && !st.dragging)
                })
                .unwrap_or(false)
        });
        if should_hide {
            hide_panel();
        }
    }

    unsafe extern "C" fn on_app_resign_active(_this: Id, _cmd: Sel, _notif: Id) {
        let should_hide = STATE.with(|cell| {
            cell.try_borrow()
                .ok()
                .and_then(|b| {
                    b.as_ref()
                        .map(|st| st.visible && !st.suppress_auto_hide && !st.dragging)
                })
                .unwrap_or(false)
        });
        if should_hide {
            hide_panel();
        }
    }

    unsafe extern "C" fn panel_can_become_key(_this: Id, _cmd: Sel) -> bool {
        true
    }

    unsafe extern "C" fn panel_can_become_main(_this: Id, _cmd: Sel) -> bool {
        true
    }

    unsafe extern "C" fn on_status_bar_clicked(_this: Id, _cmd: Sel, _sender: Id) {
        let app: Id = msg!(class("NSApplication"), "sharedApplication"; Id);
        let ev: Id = msg!(app, "currentEvent"; Id);
        let mut is_secondary = false;
        if !ev.is_null() {
            let ev_type: usize = msg!(ev, "type"; usize);
            let mods: usize = msg!(ev, "modifierFlags"; usize);
            // RightMouseDown = 3, RightMouseUp = 4, ControlFlag = 1 << 18
            if ev_type == 3 || ev_type == 4 || (mods & (1 << 18)) != 0 {
                is_secondary = true;
            }
        }
        if is_secondary {
            let (menu, btn) = STATE.with(|cell| {
                cell.borrow()
                    .as_ref()
                    .map(|s| (s.status_menu, s.status_button))
                    .unwrap_or((ptr::null_mut(), ptr::null_mut()))
            });
            if !menu.is_null() && !btn.is_null() {
                let _: bool = msg!(
                    menu,
                    "popUpMenuPositioningItem:atLocation:inView:",
                    ptr::null_mut() => Id,
                    NSPoint { x: 0.0, y: -4.0 } => NSPoint,
                    btn => Id;
                    bool
                );
            }
        } else {
            toggle_panel();
        }
    }

    unsafe extern "C" fn on_menu_toggle(_this: Id, _cmd: Sel, _sender: Id) {
        toggle_panel();
    }

    unsafe extern "C" fn on_menu_voice_prompt(_this: Id, _cmd: Sel, _sender: Id) {
        show_panel(false);
        STATE.with(|cell| {
            if let Some(st) = cell.borrow().as_ref() {
                eval_js(
                    st.webview,
                    "if (window.kronosStartVoice) window.kronosStartVoice();",
                );
            }
        });
    }

    unsafe extern "C" fn on_menu_new_chat(_this: Id, _cmd: Sel, _sender: Id) {
        show_panel(true);
    }

    unsafe extern "C" fn on_menu_open_workspace(_this: Id, _cmd: Sel, _sender: Id) {
        let url = STATE.with(|cell| {
            cell.borrow()
                .as_ref()
                .map(|s| s.base_url.clone())
                .unwrap_or_else(|| "http://127.0.0.1:18080".to_string())
        });
        hide_panel();
        open_external_url(&url);
    }

    unsafe extern "C" fn on_menu_reset_pos(_this: Id, _cmd: Sel, _sender: Id) {
        reset_panel_position();
        show_panel(false);
    }

    unsafe extern "C" fn on_menu_quit(_this: Id, _cmd: Sel, _sender: Id) {
        let app: Id = msg!(class("NSApplication"), "sharedApplication"; Id);
        let _: () = msg!(app, "terminate:", ptr::null_mut() => Id; ());
    }

    /// Intercepts Cmd+V when Finder files (PDF, code, CSV, images) are on the macOS pasteboard,
    /// otherwise delegates standard text/image paste directly to WKWebView.
    unsafe extern "C" fn on_edit_paste(_this: Id, _cmd: Sel, sender: Id) {
        let webview = STATE.with(|cell| {
            cell.borrow()
                .as_ref()
                .map(|s| s.webview)
                .unwrap_or(ptr::null_mut())
        });
        if !webview.is_null() {
            let pb: Id = msg!(class("NSPasteboard"), "generalPasteboard"; Id);
            if !pb.is_null() {
                let legacy_type = ns_string("NSFilenamesPboardType");
                let list: Id = msg!(pb, "propertyListForType:", legacy_type => Id; Id);
                if !list.is_null() {
                    let count: usize = msg!(list, "count"; usize);
                    let mut injected_any = false;
                    for i in 0..count {
                        let item: Id = msg!(list, "objectAtIndex:", i => usize; Id);
                        if let Some(p_str) = ns_string_to_rust(item) {
                            if inject_file_into_webview(webview, Path::new(&p_str)) {
                                injected_any = true;
                            }
                        }
                    }
                    if injected_any {
                        return;
                    }
                }
            }
        }
        let app: Id = msg!(class("NSApplication"), "sharedApplication"; Id);
        let _: bool = msg!(
            app,
            "sendAction:to:from:",
            sel("paste:") => Sel,
            ptr::null_mut() => Id,
            sender => Id;
            bool
        );
    }

    unsafe extern "C" fn on_script_message(_this: Id, _cmd: Sel, _ctrl: Id, message: Id) {
        if message.is_null() {
            return;
        }
        let body: Id = msg!(message, "body"; Id);
        if body.is_null() {
            return;
        }
        let dict_cls = class("NSDictionary");
        let is_dict: bool = msg!(body, "isKindOfClass:", dict_cls => Id; bool);
        if !is_dict {
            return;
        }
        let k_type = ns_string("type");
        let v_type: Id = msg!(body, "objectForKey:", k_type => Id; Id);
        let Some(msg_type) = ns_string_to_rust(v_type) else {
            return;
        };
        match msg_type.as_str() {
            "resize" => {
                let k_h = ns_string("height");
                let v_h: Id = msg!(body, "objectForKey:", k_h => Id; Id);
                if !v_h.is_null() {
                    let h: f64 = msg!(v_h, "doubleValue"; f64);
                    if h > 40.0 {
                        resize_panel(h);
                    }
                }
            }
            "hide" => {
                hide_panel();
            }
            "open_url" => {
                let k_u = ns_string("url");
                let v_u: Id = msg!(body, "objectForKey:", k_u => Id; Id);
                if let Some(url) = ns_string_to_rust(v_u) {
                    hide_panel();
                    open_external_url(&url);
                }
            }
            "pick_file" => {
                pick_files_with_open_panel();
            }
            "drag_start" => {
                let mouse: NSPoint = msg!(class("NSEvent"), "mouseLocation"; NSPoint);
                STATE.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    let Some(st) = borrow.as_mut() else { return };
                    let frame: NSRect = msg!(st.panel, "frame"; NSRect);
                    st.panel_x = frame.origin.x;
                    st.panel_top_y = frame.origin.y + frame.size.height;
                    st.dragging = true;
                    st.drag_mouse_start = (mouse.x, mouse.y);
                    st.drag_panel_start = (st.panel_x, st.panel_top_y);
                });
            }
            "drag_move" => {
                let mouse: NSPoint = msg!(class("NSEvent"), "mouseLocation"; NSPoint);
                STATE.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    let Some(st) = borrow.as_mut() else { return };
                    if !st.dragging {
                        return;
                    }
                    let next_x = (st.drag_panel_start.0 + (mouse.x - st.drag_mouse_start.0)).round();
                    let raw_top_y = (st.drag_panel_start.1 + (mouse.y - st.drag_mouse_start.1)).round();
                    let next_top_y = clamp_top_below_menubar(next_x, raw_top_y);
                    if (next_x - st.panel_x).abs() < 0.5 && (next_top_y - st.panel_top_y).abs() < 0.5 {
                        return;
                    }
                    st.panel_x = next_x;
                    st.panel_top_y = next_top_y;
                    st.has_custom_pos = true;
                    let frame = NSRect {
                        origin: NSPoint {
                            x: st.panel_x,
                            y: st.panel_top_y - st.panel_height,
                        },
                        size: NSSize {
                            width: PANEL_WIDTH,
                            height: st.panel_height,
                        },
                    };
                    let _: () = msg!(st.panel, "setFrame:display:", frame => NSRect, true => bool; ());
                });
            }
            "drag_end" => {
                STATE.with(|cell| {
                    if let Some(st) = cell.borrow_mut().as_mut() {
                        st.dragging = false;
                    }
                });
            }
            "reset_pos" => {
                reset_panel_position();
            }
            "capture_begin" => {
                STATE.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    if let Some(st) = borrow.as_mut() {
                        st.suppress_auto_hide = true;
                        let _: () = msg!(st.panel, "orderOut:", ptr::null_mut() => Id; ());
                    }
                });
            }
            "capture_end" => {
                STATE.with(|cell| {
                    if let Some(st) = cell.borrow_mut().as_mut() {
                        st.suppress_auto_hide = false;
                    }
                });
                show_panel(false);
            }
            "mic_begin" => {
                STATE.with(|cell| {
                    if let Some(st) = cell.borrow_mut().as_mut() {
                        st.suppress_auto_hide = true;
                    }
                });
            }
            "mic_end" => {
                STATE.with(|cell| {
                    if let Some(st) = cell.borrow_mut().as_mut() {
                        st.suppress_auto_hide = false;
                    }
                });
            }
            _ => {}
        }
    }

    unsafe extern "C" fn on_request_media_capture_permission(
        _this: Id,
        _cmd: Sel,
        _webview: Id,
        origin: Id,
        _frame: Id,
        _media_type: isize,
        decision_handler: *mut c_void,
    ) {
        if decision_handler.is_null() {
            return;
        }
        let mut allow = true;
        if !origin.is_null() {
            let host_ns: Id = msg!(origin, "host"; Id);
            if let Some(host) = ns_string_to_rust(host_ns) {
                let h = host.trim().to_ascii_lowercase();
                if !h.is_empty() && h != "127.0.0.1" && h != "localhost" && h != "::1" {
                    let matches_base = STATE.with(|cell| {
                        cell.borrow()
                            .as_ref()
                            .map(|s| s.base_url.to_ascii_lowercase().contains(&h))
                            .unwrap_or(false)
                    });
                    allow = matches_base;
                }
            }
        }
        // WKPermissionDecisionGrant = 1, WKPermissionDecisionDeny = 2
        let decision: isize = if allow { 1 } else { 2 };
        let block = decision_handler as *mut ObjCBlockLiteral;
        if let Some(invoke) = (*block).invoke {
            invoke(block, decision);
        }
    }

    unsafe extern "C" fn carbon_hotkey_handler(
        _next: *mut c_void,
        _event: *mut c_void,
        _user_data: *mut c_void,
    ) -> i32 {
        let pool = objc_autoreleasePoolPush();
        toggle_panel();
        objc_autoreleasePoolPop(pool);
        0
    }

    unsafe fn register_objc_classes() -> (Id, Id) {
        let mut panel_cls = class("KronosQuickPanel");
        if panel_cls.is_null() {
            let name = CString::new("KronosQuickPanel").unwrap();
            panel_cls = objc_allocateClassPair(class("NSPanel"), name.as_ptr(), 0);
            let bool_sig = CString::new("B@:").unwrap();
            class_addMethod(
                panel_cls,
                sel("canBecomeKeyWindow"),
                panel_can_become_key as *const c_void,
                bool_sig.as_ptr(),
            );
            class_addMethod(
                panel_cls,
                sel("canBecomeMainWindow"),
                panel_can_become_main as *const c_void,
                bool_sig.as_ptr(),
            );
            objc_registerClassPair(panel_cls);
        }

        let mut ctrl_cls = class("KronosBarController");
        if ctrl_cls.is_null() {
            let name = CString::new("KronosBarController").unwrap();
            ctrl_cls = objc_allocateClassPair(class("NSObject"), name.as_ptr(), 0);
            let action_sig = CString::new("v@:@").unwrap();
            let msg_sig = CString::new("v@:@@").unwrap();
            let media_perm_sig = CString::new("v@:@@@q@?").unwrap();
            class_addMethod(
                ctrl_cls,
                sel("statusBarClicked:"),
                on_status_bar_clicked as *const c_void,
                action_sig.as_ptr(),
            );
            class_addMethod(
                ctrl_cls,
                sel("toggleQuickPrompt:"),
                on_menu_toggle as *const c_void,
                action_sig.as_ptr(),
            );
            class_addMethod(
                ctrl_cls,
                sel("voiceQuickPrompt:"),
                on_menu_voice_prompt as *const c_void,
                action_sig.as_ptr(),
            );
            class_addMethod(
                ctrl_cls,
                sel("newQuickChat:"),
                on_menu_new_chat as *const c_void,
                action_sig.as_ptr(),
            );
            class_addMethod(
                ctrl_cls,
                sel("openWorkspace:"),
                on_menu_open_workspace as *const c_void,
                action_sig.as_ptr(),
            );
            class_addMethod(
                ctrl_cls,
                sel("resetHudPosition:"),
                on_menu_reset_pos as *const c_void,
                action_sig.as_ptr(),
            );
            class_addMethod(
                ctrl_cls,
                sel("quitBar:"),
                on_menu_quit as *const c_void,
                action_sig.as_ptr(),
            );
            class_addMethod(
                ctrl_cls,
                sel("smartPaste:"),
                on_edit_paste as *const c_void,
                action_sig.as_ptr(),
            );
            class_addMethod(
                ctrl_cls,
                sel("windowDidResignKey:"),
                on_window_resign_key as *const c_void,
                action_sig.as_ptr(),
            );
            class_addMethod(
                ctrl_cls,
                sel("applicationDidResignActive:"),
                on_app_resign_active as *const c_void,
                action_sig.as_ptr(),
            );
            class_addMethod(
                ctrl_cls,
                sel("userContentController:didReceiveScriptMessage:"),
                on_script_message as *const c_void,
                msg_sig.as_ptr(),
            );
            class_addMethod(
                ctrl_cls,
                sel("webView:requestMediaCapturePermissionForOrigin:initiatedByFrame:type:decisionHandler:"),
                on_request_media_capture_permission as *const c_void,
                media_perm_sig.as_ptr(),
            );
            objc_registerClassPair(ctrl_cls);
        }
        (panel_cls, ctrl_cls)
    }

    /// Attaches a standard Edit menu so Cmd+C, Cmd+V, Cmd+X, Cmd+A, and Cmd+Z work inside WKWebView.
    unsafe fn install_edit_main_menu(app: Id, controller: Id) {
        let main_menu: Id = msg!(class("NSMenu"), "new"; Id);
        let edit_container: Id = msg!(class("NSMenuItem"), "new"; Id);
        let edit_title = ns_string("Edit");
        let edit_alloc: Id = msg!(class("NSMenu"), "alloc"; Id);
        let edit_menu: Id = msg!(edit_alloc, "initWithTitle:", edit_title => Id; Id);

        let add_std = |title: &str, action: &str, key: &str, target: Id| {
            let item_alloc: Id = msg!(class("NSMenuItem"), "alloc"; Id);
            let t = ns_string(title);
            let k = ns_string(key);
            let item: Id = msg!(
                item_alloc,
                "initWithTitle:action:keyEquivalent:",
                t => Id,
                sel(action) => Sel,
                k => Id;
                Id
            );
            if !target.is_null() {
                let _: () = msg!(item, "setTarget:", target => Id; ());
            }
            let _: () = msg!(edit_menu, "addItem:", item => Id; ());
        };

        add_std("Undo", "undo:", "z", ptr::null_mut());
        add_std("Cut", "cut:", "x", ptr::null_mut());
        add_std("Copy", "copy:", "c", ptr::null_mut());
        add_std("Paste", "smartPaste:", "v", controller);
        add_std("Select All", "selectAll:", "a", ptr::null_mut());
        add_std("Quit", "quitBar:", "q", controller);

        let _: () = msg!(edit_container, "setSubmenu:", edit_menu => Id; ());
        let _: () = msg!(main_menu, "addItem:", edit_container => Id; ());
        let _: () = msg!(app, "setMainMenu:", main_menu => Id; ());
    }

    unsafe fn build_status_dropdown_menu(controller: Id) -> Id {
        let menu: Id = msg!(class("NSMenu"), "new"; Id);
        let add_item = |title: &str, action: &str, key: &str| {
            let alloc: Id = msg!(class("NSMenuItem"), "alloc"; Id);
            let t = ns_string(title);
            let k = ns_string(key);
            let item: Id = msg!(
                alloc,
                "initWithTitle:action:keyEquivalent:",
                t => Id,
                sel(action) => Sel,
                k => Id;
                Id
            );
            let _: () = msg!(item, "setTarget:", controller => Id; ());
            let _: () = msg!(menu, "addItem:", item => Id; ());
        };

        add_item("Quick Prompt (⌥Space)", "toggleQuickPrompt:", "");
        add_item("Voice Prompt (⌘⇧V)", "voiceQuickPrompt:", "");
        add_item("New Quick Chat", "newQuickChat:", "n");
        add_item("Reset HUD Position", "resetHudPosition:", "");
        add_item("Open Kronos Workspace", "openWorkspace:", "o");
        let sep: Id = msg!(class("NSMenuItem"), "separatorItem"; Id);
        let _: () = msg!(menu, "addItem:", sep => Id; ());
        add_item("Quit Kronos Bar", "quitBar:", "q");
        menu
    }

    unsafe fn register_global_hotkeys() {
        let spec = EventTypeSpec {
            event_class: u32::from_be_bytes(*b"keyb"),
            event_kind: 5, // kEventHotKeyPressed
        };
        let target = GetApplicationEventTarget();
        let mut handler_ref: *mut c_void = ptr::null_mut();
        let _ = InstallEventHandler(
            target,
            carbon_hotkey_handler,
            1,
            &spec,
            ptr::null_mut(),
            &mut handler_ref,
        );

        // Hotkey 1: Option + Space (kVK_Space = 0x31, optionKey = 1 << 11)
        let mut hk1: *mut c_void = ptr::null_mut();
        let s1 = RegisterEventHotKey(
            0x31,
            1 << 11,
            EventHotKeyID {
                signature: u32::from_be_bytes(*b"KRN1"),
                id: 1,
            },
            target,
            0,
            &mut hk1,
        );

        // Hotkey 2: Cmd + Shift + K (kVK_ANSI_K = 0x28, cmdKey | shiftKey = (1 << 8) | (1 << 9))
        let mut hk2: *mut c_void = ptr::null_mut();
        let s2 = RegisterEventHotKey(
            0x28,
            (1 << 8) | (1 << 9),
            EventHotKeyID {
                signature: u32::from_be_bytes(*b"KRN2"),
                id: 2,
            },
            target,
            0,
            &mut hk2,
        );

        eprintln!(
            "kronos_menubar global shortcuts registered (Option+Space status={s1}, Cmd+Shift+K status={s2})"
        );
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let flag = |name: &str| {
            args.iter()
                .position(|a| a == name)
                .and_then(|i| args.get(i + 1).cloned())
        };
        let base_url = flag("--url").unwrap_or_else(|| "http://127.0.0.1:18080".to_string());
        let open_now = args.iter().any(|a| a == "--show");

        if args.iter().any(|a| a == "--self-test") {
            let rgba = render_template_icon_rgba(36);
            assert_eq!(rgba.len(), 36 * 36 * 4);
            assert!(rgba.chunks_exact(4).any(|px| px[3] > 200));
            unsafe {
                let (p_cls, c_cls) = register_objc_classes();
                assert!(!p_cls.is_null() && !c_cls.is_null());
            }
            println!("kronos_menubar self-test passed");
            return;
        }

        unsafe {
            let pool = objc_autoreleasePoolPush();
            let (panel_cls, ctrl_cls) = register_objc_classes();
            let app: Id = msg!(class("NSApplication"), "sharedApplication"; Id);
            // NSApplicationActivationPolicyAccessory = 1 (top menu bar only, no Dock clutter)
            let _: bool = msg!(app, "setActivationPolicy:", 1isize => isize; bool);

            let controller: Id = msg!(ctrl_cls, "new"; Id);
            let _: () = msg!(app, "setDelegate:", controller => Id; ());
            install_edit_main_menu(app, controller);
            let status_menu = build_status_dropdown_menu(controller);

            // Top menu bar status item (NSVariableStatusItemLength = -1.0)
            let sys_bar: Id = msg!(class("NSStatusBar"), "systemStatusBar"; Id);
            let status_item: Id = msg!(sys_bar, "statusItemWithLength:", -1.0f64 => f64; Id);
            let _: Id = msg!(status_item, "retain"; Id);
            let button: Id = msg!(status_item, "button"; Id);
            if !button.is_null() {
                let icon = build_status_template_image();
                if !icon.is_null() {
                    let _: () = msg!(button, "setImage:", icon => Id; ());
                } else {
                    let fallback = ns_string("K");
                    let _: () = msg!(button, "setTitle:", fallback => Id; ());
                }
                let tip = ns_string("Kronos Quick Prompt (⌥Space or ⌘⇧K · Right-click for menu)");
                let _: () = msg!(button, "setToolTip:", tip => Id; ());
                let _: () = msg!(button, "setTarget:", controller => Id; ());
                let _: () = msg!(button, "setAction:", sel("statusBarClicked:") => Sel; ());
                // NSEventMaskLeftMouseUp (1<<2) | NSEventMaskRightMouseUp (1<<4)
                let _: isize = msg!(button, "sendActionOn:", (1usize << 2) | (1usize << 4) => usize; isize);
            }

            // Floating borderless key-capable NSPanel + transparent WKWebView
            let (ax, top_y) = compute_default_anchor();
            let init_rect = NSRect {
                origin: NSPoint {
                    x: ax,
                    y: top_y - PANEL_INIT_HEIGHT,
                },
                size: NSSize {
                    width: PANEL_WIDTH,
                    height: PANEL_INIT_HEIGHT,
                },
            };
            let panel_alloc: Id = msg!(panel_cls, "alloc"; Id);
            // styleMask = 0 (Borderless), backing = 2 (NSBackingStoreBuffered)
            let panel: Id = msg!(
                panel_alloc,
                "initWithContentRect:styleMask:backing:defer:",
                init_rect => NSRect,
                0usize => usize,
                2usize => usize,
                false => bool;
                Id
            );
            let clear: Id = msg!(class("NSColor"), "clearColor"; Id);
            let _: () = msg!(panel, "setOpaque:", false => bool; ());
            let _: () = msg!(panel, "setBackgroundColor:", clear => Id; ());
            let _: () = msg!(panel, "setHasShadow:", true => bool; ());
            let _: () = msg!(panel, "setLevel:", 3isize => isize; ()); // NSFloatingWindowLevel
            // CanJoinAllSpaces (1<<0) | FullScreenAuxiliary (1<<8)
            let _: () = msg!(panel, "setCollectionBehavior:", (1usize << 0) | (1usize << 8) => usize; ());
            let _: () = msg!(panel, "setMovableByWindowBackground:", false => bool; ());
            let _: () = msg!(panel, "setHidesOnDeactivate:", false => bool; ());
            let _: () = msg!(panel, "setDelegate:", controller => Id; ());

            let wk_cfg: Id = msg!(class("WKWebViewConfiguration"), "new"; Id);
            let ucc: Id = msg!(wk_cfg, "userContentController"; Id);
            let handler_name = ns_string("kronos");
            let _: () = msg!(
                ucc,
                "addScriptMessageHandler:name:",
                controller => Id,
                handler_name => Id;
                ()
            );

            let view_rect = NSRect {
                origin: NSPoint { x: 0.0, y: 0.0 },
                size: NSSize {
                    width: PANEL_WIDTH,
                    height: PANEL_INIT_HEIGHT,
                },
            };
            let wk_alloc: Id = msg!(class("WKWebView"), "alloc"; Id);
            let webview: Id = msg!(
                wk_alloc,
                "initWithFrame:configuration:",
                view_rect => NSRect,
                wk_cfg => Id;
                Id
            );
            // NSViewWidthSizable (2) | NSViewHeightSizable (16)
            let _: () = msg!(webview, "setAutoresizingMask:", 18usize => usize; ());
            let _: () = msg!(webview, "setUIDelegate:", controller => Id; ());
            let false_num: Id = msg!(class("NSNumber"), "numberWithBool:", false => bool; Id);
            let draws_bg_key = ns_string("drawsBackground");
            let _: () = msg!(webview, "setValue:forKey:", false_num => Id, draws_bg_key => Id; ());

            let quick_url = format!("{}/quick", base_url.trim_end_matches('/'));
            let ns_u = ns_string(&quick_url);
            let url_obj: Id = msg!(class("NSURL"), "URLWithString:", ns_u => Id; Id);
            if !url_obj.is_null() {
                let req: Id = msg!(
                    class("NSURLRequest"),
                    "requestWithURL:cachePolicy:timeoutInterval:",
                    url_obj => Id,
                    1usize => usize,
                    15.0f64 => f64;
                    Id
                );
                let _: Id = msg!(webview, "loadRequest:", req => Id; Id);
            }

            // Rounded CALayer container eliminates rectangular borderless window corner rim on macOS 14+.
            let container_alloc: Id = msg!(class("NSView"), "alloc"; Id);
            let container: Id = msg!(container_alloc, "initWithFrame:", view_rect => NSRect; Id);
            let _: () = msg!(container, "setAutoresizingMask:", 18usize => usize; ());
            let _: () = msg!(container, "setWantsLayer:", true => bool; ());
            let layer: Id = msg!(container, "layer"; Id);
            if !layer.is_null() {
                let _: () = msg!(layer, "setCornerRadius:", 18.0f64 => f64; ());
                let _: () = msg!(layer, "setMasksToBounds:", true => bool; ());
            }
            let _: () = msg!(container, "addSubview:", webview => Id; ());
            let _: () = msg!(panel, "setContentView:", container => Id; ());

            STATE.with(|cell| {
                *cell.borrow_mut() = Some(BarState {
                    base_url: base_url.clone(),
                    panel,
                    webview,
                    status_button: button,
                    status_menu,
                    panel_x: ax,
                    panel_top_y: top_y,
                    panel_height: PANEL_INIT_HEIGHT,
                    has_custom_pos: false,
                    dragging: false,
                    suppress_auto_hide: false,
                    drag_mouse_start: (0.0, 0.0),
                    drag_panel_start: (ax, top_y),
                    visible: false,
                });
            });

            register_global_hotkeys();
            eprintln!("kronos_menubar active in macOS menu bar -> {quick_url}");

            if open_now {
                show_panel(false);
            }

            objc_autoreleasePoolPop(pool);
            let _: () = msg!(app, "run"; ());
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_template_icon_rendering_and_classes() {
            let rgba = render_template_icon_rgba(36);
            assert_eq!(rgba.len(), 36 * 36 * 4);
            let opaque = rgba.chunks_exact(4).filter(|p| p[3] > 180).count();
            let transparent = rgba.chunks_exact(4).filter(|p| p[3] == 0).count();
            assert!(opaque > 80, "expected visible starburst rays, got {opaque}");
            assert!(transparent > 400, "expected transparent background, got {transparent}");
            unsafe {
                let (p_cls, c_cls) = register_objc_classes();
                assert!(!p_cls.is_null());
                assert!(!c_cls.is_null());
            }
        }

        #[test]
        fn test_anchor_visibility_and_menubar_clamp() {
            let (ax, top_y) = compute_default_anchor();
            assert!(top_y > 180.0);
            unsafe {
                assert!(is_anchor_visible_on_any_screen(ax, top_y));
                assert!(!is_anchor_visible_on_any_screen(-99_999.0, -99_999.0));
                let clamped = clamp_top_below_menubar(ax, top_y + 10_000.0);
                assert!(clamped <= top_y + 10_000.0);
            }
        }
    }
}
