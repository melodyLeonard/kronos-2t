//! Zero-crate universal file attachment ingestion: PDF (PDFKit + Vision OCR fallback), RTF/DOCX, code, CSV, ZIP, and images.

use crate::runtime::agent::truncate_utf8;
use crate::runtime::vision::{analyze_image_bytes, decode_base64};
use std::path::Path;
use std::time::Instant;

const MAX_INLINE_TEXT_BYTES: usize = 16 * 1024;

#[derive(Clone, Debug)]
pub struct AttachmentIngestion {
    pub kind: String,
    pub name: String,
    pub badge: String,
    pub prompt_context: String,
    pub saved_rel_path: Option<String>,
    pub is_image: bool,
    pub elapsed_ms: f64,
}

fn sanitize_filename(raw: &str, fallback_ext: &str) -> String {
    let base = raw
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("attachment")
        .trim();
    let mut clean = String::with_capacity(base.len().min(64));
    for ch in base.chars().take(64) {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
            clean.push(ch);
        } else if ch.is_whitespace() {
            clean.push('_');
        }
    }
    while clean.starts_with('.') {
        clean.remove(0);
    }
    if clean.is_empty() {
        clean = "attachment".to_string();
    }
    if !clean.contains('.') && !fallback_ext.is_empty() {
        clean.push('.');
        clean.push_str(fallback_ext);
    }
    clean
}

fn ext_from_name_or_mime(name: &str, mime: &str, bytes: &[u8]) -> String {
    if bytes.starts_with(b"%PDF-") {
        return "pdf".to_string();
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return "png".to_string();
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return "jpg".to_string();
    }
    if bytes.starts_with(b"GIF8") {
        return "gif".to_string();
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return "webp".to_string();
    }
    if bytes.starts_with(b"{\\rtf") {
        return "rtf".to_string();
    }
    let name_ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !name_ext.is_empty() {
        return name_ext;
    }
    if bytes.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
        if mime.contains("wordprocessingml") {
            return "docx".to_string();
        }
        return "zip".to_string();
    }
    match mime.to_ascii_lowercase().as_str() {
        "application/pdf" => "pdf".to_string(),
        "image/png" => "png".to_string(),
        "image/jpeg" | "image/jpg" => "jpg".to_string(),
        "image/webp" => "webp".to_string(),
        "image/gif" => "gif".to_string(),
        "text/csv" => "csv".to_string(),
        "application/json" => "json".to_string(),
        "text/markdown" => "md".to_string(),
        "text/html" => "html".to_string(),
        "application/rtf" | "text/rtf" => "rtf".to_string(),
        _ => "txt".to_string(),
    }
}

fn is_image_ext(ext: &str) -> bool {
    matches!(
        ext,
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tiff" | "tif" | "heic" | "ico"
    )
}

fn decode_data_or_base64(payload: &str) -> Result<(String, Vec<u8>), String> {
    let trimmed = payload.trim();
    if let Some(rest) = trimmed.strip_prefix("data:") {
        let Some((meta, body)) = rest.split_once(',') else {
            return Err("malformed data URL in attachment".to_string());
        };
        let mime = meta.split(';').next().unwrap_or("").to_string();
        if meta.contains(";base64") {
            let bytes = decode_base64(body)?;
            Ok((mime, bytes))
        } else {
            Ok((mime, body.as_bytes().to_vec()))
        }
    } else {
        let bytes = decode_base64(trimmed)?;
        Ok((String::new(), bytes))
    }
}

pub fn ingest_attachment(
    raw_name: &str,
    hint_mime: &str,
    payload: &str,
    workdir: Option<&Path>,
    slot: usize,
) -> Result<AttachmentIngestion, String> {
    let (url_mime, bytes) = decode_data_or_base64(payload)?;
    let mime = if !hint_mime.is_empty() {
        hint_mime
    } else {
        url_mime.as_str()
    };
    ingest_attachment_bytes(raw_name, mime, &bytes, workdir, slot)
}

pub fn ingest_attachment_bytes(
    raw_name: &str,
    mime: &str,
    bytes: &[u8],
    workdir: Option<&Path>,
    slot: usize,
) -> Result<AttachmentIngestion, String> {
    let t0 = Instant::now();
    let ext = ext_from_name_or_mime(raw_name, mime, bytes);
    let default_name = format!("attachment_{slot}.{ext}");
    let safe_name = sanitize_filename(
        if raw_name.trim().is_empty() {
            &default_name
        } else {
            raw_name
        },
        &ext,
    );

    if is_image_ext(&ext) {
        let img = analyze_image_bytes(bytes, &ext, workdir, slot)?;
        return Ok(AttachmentIngestion {
            kind: "image".to_string(),
            name: safe_name,
            badge: format!("vision · {}", img.short_badge()),
            prompt_context: img.to_prompt_context(slot),
            saved_rel_path: img.saved_rel_path,
            is_image: true,
            elapsed_ms: img.elapsed_ms,
        });
    }

    let saved_rel_path = if let Some(dir) = workdir {
        let media_dir = dir.join(".kronos_media");
        if std::fs::create_dir_all(&media_dir).is_ok() {
            let fname = format!("file_{slot}_{safe_name}");
            let full = media_dir.join(&fname);
            if std::fs::write(&full, bytes).is_ok() {
                Some(format!(".kronos_media/{fname}"))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };
    let truncated_saved_note = match &saved_rel_path {
        Some(rel) => format!(", full_file_workspace_path=\"{rel}\""),
        None => String::new(),
    };

    if ext == "pdf" {
        let (pages, ocr_pages, extracted) = extract_pdf_text(bytes);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ocr_tag = if ocr_pages > 0 {
            format!(" ({ocr_pages} OCR)")
        } else {
            String::new()
        };
        let char_count = extracted.chars().count();
        let badge = format!(
            "pdf · {safe_name} · {pages}p{ocr_tag} · {} chars · {:.0} ms",
            char_count, elapsed_ms
        );
        let body = if extracted.trim().is_empty() {
            "(PDF contained no extractable text)".to_string()
        } else {
            truncate_utf8(&extracted, MAX_INLINE_TEXT_BYTES)
        };
        let prompt_context = format!(
            "[Attached PDF #{slot}: \"{safe_name}\", {pages} page(s){ocr_tag}, {} bytes, extracted text inline below]\n{body}\n",
            bytes.len()
        );
        return Ok(AttachmentIngestion {
            kind: "pdf".to_string(),
            name: safe_name,
            badge,
            prompt_context,
            saved_rel_path,
            is_image: false,
            elapsed_ms,
        });
    }

    if matches!(ext.as_str(), "rtf" | "docx" | "doc" | "odt") {
        let extracted = extract_rich_document_text(bytes, &ext);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let lines = extracted.lines().count();
        let badge = format!(
            "doc · {safe_name} · {lines} lines · {:.0} ms",
            elapsed_ms
        );
        let body = if extracted.trim().is_empty() {
            "(Document contained no extractable plain text)".to_string()
        } else {
            truncate_utf8(&extracted, MAX_INLINE_TEXT_BYTES)
        };
        let prompt_context = format!(
            "[Attached Document #{slot}: \"{safe_name}\", format={}, {} bytes, extracted text inline below]\n{body}\n",
            ext.to_uppercase(),
            bytes.len()
        );
        return Ok(AttachmentIngestion {
            kind: "document".to_string(),
            name: safe_name,
            badge,
            prompt_context,
            saved_rel_path,
            is_image: false,
            elapsed_ms,
        });
    }

    if ext == "zip" {
        let entries = list_zip_entries(bytes);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let badge = format!(
            "archive · {safe_name} · {} entries · {} B",
            entries.len(),
            bytes.len()
        );
        let listing = if entries.is_empty() {
            "(empty or unreadable ZIP archive)".to_string()
        } else {
            entries
                .iter()
                .take(80)
                .map(|(n, sz)| format!("  - {n} ({sz} B)"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let prompt_context = format!(
            "[Attached Archive #{slot}: \"{safe_name}\", {} entries, {} bytes]\nArchive contents:\n{listing}\n",
            entries.len(),
            bytes.len()
        );
        return Ok(AttachmentIngestion {
            kind: "archive".to_string(),
            name: safe_name,
            badge,
            prompt_context,
            saved_rel_path,
            is_image: false,
            elapsed_ms,
        });
    }

    if let Ok(utf8) = std::str::from_utf8(bytes) {
        let line_count = utf8.lines().count();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let extra_meta = if ext == "csv" || ext == "tsv" {
            let header = utf8.lines().next().unwrap_or("");
            let cols = if ext == "tsv" {
                header.split('\t').count()
            } else {
                header.split(',').count()
            };
            format!(", rows={line_count}, cols={cols}")
        } else {
            format!(", lines={line_count}")
        };
        let badge = format!("file · {safe_name} · {line_count} lines · {} B", bytes.len());
        let path_note = if utf8.len() > MAX_INLINE_TEXT_BYTES {
            truncated_saved_note.as_str()
        } else {
            ""
        };
        let body = truncate_utf8(utf8, MAX_INLINE_TEXT_BYTES);
        let prompt_context = format!(
            "[Attached File #{slot}: \"{safe_name}\", type={}{extra_meta}, {} bytes{path_note}]\n```{ext}\n{body}\n```\n",
            ext.to_uppercase(),
            bytes.len()
        );
        return Ok(AttachmentIngestion {
            kind: "file".to_string(),
            name: safe_name,
            badge,
            prompt_context,
            saved_rel_path,
            is_image: false,
            elapsed_ms,
        });
    }

    let saved_note = truncated_saved_note;

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let hex_head: String = bytes
        .iter()
        .take(32)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ");
    let badge = format!("binary · {safe_name} · {} B", bytes.len());
    let prompt_context = format!(
        "[Attached Binary File #{slot}: \"{safe_name}\", {} bytes{saved_note}]\nHeader hex: {hex_head}\n",
        bytes.len()
    );
    Ok(AttachmentIngestion {
        kind: "binary".to_string(),
        name: safe_name,
        badge,
        prompt_context,
        saved_rel_path,
        is_image: false,
        elapsed_ms,
    })
}

fn list_zip_entries(bytes: &[u8]) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 30 <= bytes.len() {
        if &bytes[i..i + 4] == b"PK\x03\x04" {
            let comp_size =
                u32::from_le_bytes([bytes[i + 18], bytes[i + 19], bytes[i + 20], bytes[i + 21]])
                    as usize;
            let uncomp_size =
                u32::from_le_bytes([bytes[i + 22], bytes[i + 23], bytes[i + 24], bytes[i + 25]]);
            let name_len = u16::from_le_bytes([bytes[i + 26], bytes[i + 27]]) as usize;
            let extra_len = u16::from_le_bytes([bytes[i + 28], bytes[i + 29]]) as usize;
            let name_start = i + 30;
            if name_start + name_len <= bytes.len() {
                let name = String::from_utf8_lossy(&bytes[name_start..name_start + name_len])
                    .into_owned();
                if !name.ends_with('/') {
                    out.push((name, uncomp_size));
                }
            }
            let next = name_start
                .saturating_add(name_len)
                .saturating_add(extra_len)
                .saturating_add(comp_size);
            if next <= i {
                break;
            }
            i = next;
        } else {
            i += 1;
        }
    }
    out
}

fn fallback_pdf_literal_strings(bytes: &[u8]) -> (usize, String) {
    let raw = String::from_utf8_lossy(bytes);
    let pages = raw.matches("/Type /Page").count().saturating_sub(raw.matches("/Type /Pages").count()).max(1);
    let mut out = String::new();
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '(' {
            let mut depth = 1usize;
            let mut esc = false;
            let mut s = String::new();
            for ch in chars.by_ref() {
                if esc {
                    esc = false;
                    match ch {
                        'n' => s.push('\n'),
                        'r' => s.push('\r'),
                        't' => s.push('\t'),
                        other => s.push(other),
                    }
                    continue;
                }
                if ch == '\\' {
                    esc = true;
                } else if ch == '(' {
                    depth += 1;
                    s.push('(');
                } else if ch == ')' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    s.push(')');
                } else {
                    s.push(ch);
                }
            }
            let trimmed = s.trim();
            if trimmed.len() >= 2 && trimmed.chars().all(|x| !x.is_control() || x.is_whitespace()) {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(trimmed);
            }
        }
    }
    (pages, out)
}

fn extract_pdf_text(bytes: &[u8]) -> (usize, usize, String) {
    #[cfg(target_os = "macos")]
    {
        if let Some(res) = macos::extract_pdf_macos(bytes) {
            return res;
        }
    }
    let (pages, txt) = fallback_pdf_literal_strings(bytes);
    (pages, 0, txt)
}

fn extract_rich_document_text(bytes: &[u8], _ext: &str) -> String {
    #[cfg(target_os = "macos")]
    {
        if let Some(s) = macos::extract_attributed_doc_macos(bytes) {
            if !s.trim().is_empty() {
                return s;
            }
        }
    }
    String::from_utf8_lossy(bytes).into_owned()
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::{c_char, c_void, CStr, CString};

    type Id = *mut c_void;
    type Sel = *mut c_void;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CGPoint {
        x: f64,
        y: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CGSize {
        width: f64,
        height: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CGRect {
        origin: CGPoint,
        size: CGSize,
    }

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
        fn objc_autoreleasePoolPush() -> *mut c_void;
        fn objc_autoreleasePoolPop(pool: *mut c_void);
    }

    #[link(name = "Foundation", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "PDFKit", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "Vision", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFDataCreate(allocator: *const c_void, bytes: *const u8, length: isize) -> *const c_void;
        fn CFRelease(cf: *const c_void);
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGDataProviderCreateWithCFData(data: *const c_void) -> *const c_void;
        fn CGDataProviderRelease(provider: *const c_void);
        fn CGPDFDocumentCreateWithProvider(provider: *const c_void) -> *const c_void;
        fn CGPDFDocumentRelease(doc: *const c_void);
        fn CGPDFDocumentGetPage(doc: *const c_void, page_number: usize) -> *const c_void;
        fn CGPDFPageGetBoxRect(page: *const c_void, box_type: i32) -> CGRect;
        fn CGColorSpaceCreateDeviceRGB() -> *const c_void;
        fn CGColorSpaceRelease(space: *const c_void);
        fn CGBitmapContextCreate(
            data: *mut u8,
            width: usize,
            height: usize,
            bits_per_component: usize,
            bytes_per_row: usize,
            space: *const c_void,
            bitmap_info: u32,
        ) -> *const c_void;
        fn CGContextSetRGBFillColor(c: *const c_void, r: f64, g: f64, b: f64, a: f64);
        fn CGContextFillRect(c: *const c_void, rect: CGRect);
        fn CGContextScaleCTM(c: *const c_void, sx: f64, sy: f64);
        fn CGContextDrawPDFPage(c: *const c_void, page: *const c_void);
        fn CGBitmapContextCreateImage(c: *const c_void) -> *const c_void;
        fn CGContextRelease(c: *const c_void);
        fn CGImageRelease(image: *const c_void);
    }

    fn sel(name: &str) -> Sel {
        let c = CString::new(name).unwrap();
        // SAFETY: NUL-terminated selector literal.
        unsafe { sel_registerName(c.as_ptr()) }
    }

    fn class(name: &str) -> Id {
        let c = CString::new(name).unwrap();
        // SAFETY: NUL-terminated class literal.
        unsafe { objc_getClass(c.as_ptr()) }
    }

    macro_rules! msg {
        ($obj:expr, $sel:expr $(, $arg:expr => $ty:ty)* ; $ret:ty) => {{
            let f: unsafe extern "C" fn(Id, Sel $(, $ty)*) -> $ret =
                std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            f($obj, sel($sel) $(, $arg)*)
        }};
    }

    pub(super) fn extract_pdf_macos(bytes: &[u8]) -> Option<(usize, usize, String)> {
        // SAFETY: Objective-C and CoreGraphics objects stay within this autorelease pool and are released before return.
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let res = extract_pdf_inner(bytes);
            objc_autoreleasePoolPop(pool);
            res
        }
    }

    unsafe fn extract_pdf_inner(bytes: &[u8]) -> Option<(usize, usize, String)> {
        let pdf_cls = class("PDFDocument");
        let data_cls = class("NSData");
        if pdf_cls.is_null() || data_cls.is_null() {
            return None;
        }
        unsafe {
            let ns_data: Id = msg!(
                data_cls,
                "dataWithBytes:length:",
                bytes.as_ptr() => *const u8,
                bytes.len() => usize;
                Id
            );
            if ns_data.is_null() {
                return None;
            }
            let doc_alloc: Id = msg!(pdf_cls, "alloc"; Id);
            let doc: Id = msg!(doc_alloc, "initWithData:", ns_data => Id; Id);
            if doc.is_null() {
                return None;
            }

            let page_count: usize = msg!(doc, "pageCount"; usize);
            let mut out = String::new();
            let mut ocr_pages = 0usize;
            let mut cg_doc: *const c_void = std::ptr::null();
            let mut cg_tried = false;

            let max_pages = page_count.min(40);
            for p in 0..max_pages {
                let page: Id = msg!(doc, "pageAtIndex:", p => usize; Id);
                let mut page_text = String::new();
                if !page.is_null() {
                    let ns_str: Id = msg!(page, "string"; Id);
                    if !ns_str.is_null() {
                        let c_ptr: *const c_char = msg!(ns_str, "UTF8String"; *const c_char);
                        if !c_ptr.is_null() {
                            page_text = CStr::from_ptr(c_ptr).to_string_lossy().trim().to_string();
                        }
                    }
                }

                // Scanned PDF page fallback: rasterize via CGPDFDocument and run Apple Vision OCR.
                let non_ws = page_text.chars().filter(|c| !c.is_whitespace()).count();
                if non_ws < 20 && ocr_pages < 6 {
                    if !cg_tried {
                        cg_tried = true;
                        let cf_data = CFDataCreate(std::ptr::null(), bytes.as_ptr(), bytes.len() as isize);
                        if !cf_data.is_null() {
                            let provider = CGDataProviderCreateWithCFData(cf_data);
                            CFRelease(cf_data);
                            if !provider.is_null() {
                                cg_doc = CGPDFDocumentCreateWithProvider(provider);
                                CGDataProviderRelease(provider);
                            }
                        }
                    }
                    if !cg_doc.is_null() {
                        if let Some(ocr_txt) = ocr_pdf_page(cg_doc, p + 1) {
                            if !ocr_txt.trim().is_empty() {
                                page_text = ocr_txt;
                                ocr_pages += 1;
                            }
                        }
                    }
                }

                if !page_text.is_empty() {
                    out.push_str(&format!("--- Page {}/{} ---\n{}\n\n", p + 1, page_count, page_text));
                }
            }

            if !cg_doc.is_null() {
                CGPDFDocumentRelease(cg_doc);
            }
            if page_count > max_pages {
                out.push_str(&format!(
                    "... ({} additional pages omitted from inline preview; full PDF saved in workspace)\n",
                    page_count - max_pages
                ));
            }
            let _: () = msg!(doc, "release"; ());
            Some((page_count, ocr_pages, out))
        }
    }

    unsafe fn ocr_pdf_page(cg_doc: *const c_void, page_num_1based: usize) -> Option<String> {
        unsafe {
            let cg_page = CGPDFDocumentGetPage(cg_doc, page_num_1based);
            if cg_page.is_null() {
                return None;
            }
            // kCGPDFMediaBox = 0.
            let rect = CGPDFPageGetBoxRect(cg_page, 0);
            if rect.size.width <= 1.0 || rect.size.height <= 1.0 {
                return None;
            }
            let scale = (1280.0 / rect.size.width.max(rect.size.height)).clamp(1.0, 2.5);
            let w = ((rect.size.width * scale).round() as usize).max(16);
            let h = ((rect.size.height * scale).round() as usize).max(16);
            let mut rgba = vec![255u8; w * h * 4];
            let cs = CGColorSpaceCreateDeviceRGB();
            let ctx = CGBitmapContextCreate(rgba.as_mut_ptr(), w, h, 8, w * 4, cs, 0x4001);
            CGColorSpaceRelease(cs);
            if ctx.is_null() {
                return None;
            }
            CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 1.0);
            CGContextFillRect(
                ctx,
                CGRect {
                    origin: CGPoint { x: 0.0, y: 0.0 },
                    size: CGSize {
                        width: w as f64,
                        height: h as f64,
                    },
                },
            );
            CGContextScaleCTM(ctx, scale, scale);
            CGContextDrawPDFPage(ctx, cg_page);
            let cg_img = CGBitmapContextCreateImage(ctx);
            CGContextRelease(ctx);
            if cg_img.is_null() {
                return None;
            }
            let text = run_vision_ocr_lines(cg_img);
            CGImageRelease(cg_img);
            Some(text)
        }
    }

    unsafe fn run_vision_ocr_lines(cg_img: *const c_void) -> String {
        let handler_cls = class("VNImageRequestHandler");
        let req_cls = class("VNRecognizeTextRequest");
        if handler_cls.is_null() || req_cls.is_null() {
            return String::new();
        }
        let mut lines: Vec<(f32, f32, String)> = Vec::new();
        unsafe {
            let empty_dict: Id = msg!(class("NSDictionary"), "dictionary"; Id);
            let handler: Id = msg!(
                msg!(handler_cls, "alloc"; Id),
                "initWithCGImage:options:",
                cg_img => *const c_void,
                empty_dict => Id;
                Id
            );
            if handler.is_null() {
                return String::new();
            }
            let req: Id = msg!(msg!(req_cls, "alloc"; Id), "init"; Id);
            if !req.is_null() {
                let _: () = msg!(req, "setRecognitionLevel:", 0isize => isize; ());
                let reqs = [req];
                let arr: Id = msg!(
                    class("NSArray"),
                    "arrayWithObjects:count:",
                    reqs.as_ptr() => *const Id,
                    1usize => usize;
                    Id
                );
                let mut err: Id = std::ptr::null_mut();
                let ok: bool = msg!(
                    handler,
                    "performRequests:error:",
                    arr => Id,
                    &mut err as *mut Id => *mut Id;
                    bool
                );
                if ok {
                    let results: Id = msg!(req, "results"; Id);
                    if !results.is_null() {
                        let n: usize = msg!(results, "count"; usize);
                        for i in 0..n {
                            let obs: Id = msg!(results, "objectAtIndex:", i => usize; Id);
                            if obs.is_null() {
                                continue;
                            }
                            let bbox: CGRect = msg!(obs, "boundingBox"; CGRect);
                            let cands: Id = msg!(obs, "topCandidates:", 1usize => usize; Id);
                            if cands.is_null() || msg!(cands, "count"; usize) == 0 {
                                continue;
                            }
                            let cand: Id = msg!(cands, "objectAtIndex:", 0usize => usize; Id);
                            let ns_str: Id = msg!(cand, "string"; Id);
                            if ns_str.is_null() {
                                continue;
                            }
                            let c_ptr: *const c_char = msg!(ns_str, "UTF8String"; *const c_char);
                            if c_ptr.is_null() {
                                continue;
                            }
                            let s = CStr::from_ptr(c_ptr).to_string_lossy().trim().to_string();
                            if !s.is_empty() {
                                let top_y = (1.0 - (bbox.origin.y + bbox.size.height)) as f32;
                                lines.push((top_y, bbox.origin.x as f32, s));
                            }
                        }
                    }
                }
                let _: () = msg!(req, "release"; ());
            }
            let _: () = msg!(handler, "release"; ());
        }
        lines.sort_by(|a, b| {
            let ra = (a.0 * 30.0).round() as i32;
            let rb = (b.0 * 30.0).round() as i32;
            ra.cmp(&rb)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        lines.into_iter().map(|(_, _, s)| s).collect::<Vec<_>>().join("\n")
    }

    pub(super) fn extract_attributed_doc_macos(bytes: &[u8]) -> Option<String> {
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let attr_cls = class("NSAttributedString");
            let data_cls = class("NSData");
            let mut out = None;
            if !attr_cls.is_null() && !data_cls.is_null() {
                let ns_data: Id = msg!(
                    data_cls,
                    "dataWithBytes:length:",
                    bytes.as_ptr() => *const u8,
                    bytes.len() => usize;
                    Id
                );
                if !ns_data.is_null() {
                    let empty_opts: Id = msg!(class("NSDictionary"), "dictionary"; Id);
                    let mut doc_attrs: Id = std::ptr::null_mut();
                    let mut err: Id = std::ptr::null_mut();
                    let alloc: Id = msg!(attr_cls, "alloc"; Id);
                    let attr: Id = msg!(
                        alloc,
                        "initWithData:options:documentAttributes:error:",
                        ns_data => Id,
                        empty_opts => Id,
                        &mut doc_attrs as *mut Id => *mut Id,
                        &mut err as *mut Id => *mut Id;
                        Id
                    );
                    if !attr.is_null() {
                        let ns_str: Id = msg!(attr, "string"; Id);
                        if !ns_str.is_null() {
                            let c_ptr: *const c_char = msg!(ns_str, "UTF8String"; *const c_char);
                            if !c_ptr.is_null() {
                                out = Some(CStr::from_ptr(c_ptr).to_string_lossy().into_owned());
                            }
                        }
                        let _: () = msg!(attr, "release"; ());
                    }
                }
            }
            objc_autoreleasePoolPop(pool);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_b64(bytes: &[u8]) -> String {
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

    fn build_valid_single_page_pdf(text: &str) -> Vec<u8> {
        let stream_body = format!("BT /F1 18 Tf 50 720 Td ({text}) Tj ET");
        let objs = [
            "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n".to_string(),
            "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n".to_string(),
            "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n".to_string(),
            format!(
                "4 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n",
                stream_body.len(),
                stream_body
            ),
            "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".to_string(),
        ];
        let mut out = Vec::from(&b"%PDF-1.4\n"[..]);
        let mut offsets = Vec::with_capacity(objs.len());
        for obj in &objs {
            offsets.push(out.len());
            out.extend_from_slice(obj.as_bytes());
        }
        let xref_pos = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
        for off in offsets {
            out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n",
                objs.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    #[test]
    fn test_pdf_rtf_csv_and_code_attachment_ingestion() {
        let tmp = std::env::temp_dir().join(format!("kronos_attach_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let pdf_bytes = build_valid_single_page_pdf("Kronos PDF Extraction Benchmark 9876");
        let pdf_url = format!("data:application/pdf;base64,{}", encode_b64(&pdf_bytes));
        let pdf_ing =
            ingest_attachment("report.pdf", "application/pdf", &pdf_url, Some(&tmp), 1).unwrap();
        assert_eq!(pdf_ing.kind, "pdf");
        assert!(!pdf_ing.is_image);
        assert!(
            pdf_ing
                .prompt_context
                .contains("Kronos PDF Extraction Benchmark 9876"),
            "unexpected pdf prompt context: {}",
            pdf_ing.prompt_context
        );
        assert!(tmp
            .join(pdf_ing.saved_rel_path.as_deref().unwrap())
            .is_file());

        let csv_raw = "model,tok_per_sec,accept_pct\nqwen-14b-spec,39.4,78.2\n";
        let csv_url = format!("data:text/csv;base64,{}", encode_b64(csv_raw.as_bytes()));
        let csv_ing = ingest_attachment("metrics.csv", "text/csv", &csv_url, Some(&tmp), 2).unwrap();
        assert_eq!(csv_ing.kind, "file");
        assert!(csv_ing.prompt_context.contains("rows=2, cols=3"));
        assert!(csv_ing.prompt_context.contains("qwen-14b-spec,39.4,78.2"));

        let rtf_raw = b"{\\rtf1\\ansi Kronos RTF Document Test 55}";
        let rtf_url = format!("data:application/rtf;base64,{}", encode_b64(rtf_raw));
        let rtf_ing =
            ingest_attachment("notes.rtf", "application/rtf", &rtf_url, Some(&tmp), 3).unwrap();
        assert_eq!(rtf_ing.kind, "document");
        assert!(rtf_ing.prompt_context.contains("Kronos RTF Document Test 55"));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
