//! Zero-crate multimodal image decoding, spatial palette extraction, and Apple Vision OCR/scene analysis.

use std::path::Path;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct OcrRegion {
    pub zone: String,
    pub text: String,
    pub top_y: f32,
    pub left_x: f32,
    pub width_n: f32,
    pub height_n: f32,
    pub bg_rgb: [u8; 3],
    pub has_left_quote_bar: bool,
    pub has_left_tick_icon: bool,
}

#[derive(Clone, Debug)]
pub struct ImagePerception {
    pub width: usize,
    pub height: usize,
    pub format_ext: String,
    pub theme: String,
    pub bg_hex: String,
    pub accent_hex: String,
    pub palette: Vec<(String, u32)>,
    pub band_colors: [String; 3],
    pub scene_tags: Vec<(String, u32)>,
    pub ocr_regions: Vec<OcrRegion>,
    pub saved_rel_path: Option<String>,
    pub elapsed_ms: f64,
}

impl ImagePerception {
    pub fn short_badge(&self) -> String {
        format!(
            "{}x{} {} · {} {} · {} text spans · {:.0} ms",
            self.width,
            self.height,
            self.theme,
            self.bg_hex,
            self.accent_hex,
            self.ocr_regions.len(),
            self.elapsed_ms
        )
    }

    pub fn to_prompt_context(&self, index: usize) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "[Attached Image #{index}: {}x{} {}, theme={}, background={}, accent={}",
            self.width,
            self.height,
            self.format_ext.to_uppercase(),
            self.theme,
            self.bg_hex,
            self.accent_hex
        ));
        if let Some(rel) = &self.saved_rel_path {
            out.push_str(&format!(", read_only_reference_image=\"{rel}\""));
        }
        out.push_str("]\n");

        if !self.palette.is_empty() {
            let pal: Vec<String> = self
                .palette
                .iter()
                .map(|(hex, pct)| format!("{hex} ({pct}%)"))
                .collect();
            out.push_str(&format!(
                "- Dominant CSS palette: {} | vertical bands top={} mid={} bottom={}\n",
                pal.join(", "),
                self.band_colors[0],
                self.band_colors[1],
                self.band_colors[2]
            ));
        }

        if !self.scene_tags.is_empty() {
            let tags: Vec<String> = self
                .scene_tags
                .iter()
                .map(|(t, pct)| format!("{t} ({pct}%)"))
                .collect();
            out.push_str(&format!("- Visual categories: {}\n", tags.join(", ")));
        }

        if self.ocr_regions.is_empty() {
            out.push_str("- Detected text (OCR): none\n");
        } else {
            out.push_str("- Detected text in 2D reading order (top -> bottom):\n");
            for reg in self.ocr_regions.iter().take(60) {
                out.push_str(&format!("  [{}] {}\n", reg.zone, reg.text));
            }
            if self.ocr_regions.len() > 60 {
                out.push_str(&format!(
                    "  ... ({} additional text spans omitted)\n",
                    self.ocr_regions.len() - 60
                ));
            }
        }
        out.push_str("[/Attached Image]\n");
        out
    }
}

/// Decodes a `data:image/...;base64,...` URL or raw Base64 payload into `(ext, raw_bytes)`.
pub fn decode_image_payload(input: &str) -> Result<(String, Vec<u8>), String> {
    let trimmed = input.trim();
    let (hint_ext, b64_part) = if let Some(rest) = trimmed.strip_prefix("data:") {
        let Some((meta, payload)) = rest.split_once(',') else {
            return Err("malformed data URL: missing comma separator".to_string());
        };
        let mime = meta.split(';').next().unwrap_or("");
        let ext = match mime {
            "image/png" => "png",
            "image/jpeg" | "image/jpg" => "jpg",
            "image/webp" => "webp",
            "image/gif" => "gif",
            _ => "",
        };
        (ext, payload)
    } else {
        ("", trimmed)
    };

    let bytes = decode_base64(b64_part)?;
    if bytes.len() < 8 {
        return Err("image payload too short".to_string());
    }
    let ext = if !hint_ext.is_empty() {
        hint_ext.to_string()
    } else {
        sniff_image_ext(&bytes).to_string()
    };
    Ok((ext, bytes))
}

fn sniff_image_ext(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        "png"
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "jpg"
    } else if bytes.starts_with(b"GIF8") {
        "gif"
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        "webp"
    } else {
        "png"
    }
}

pub fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u8;
    for b in input.bytes() {
        if b.is_ascii_whitespace() || b == b'=' {
            continue;
        }
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return Err(format!("invalid base64 byte 0x{b:02x}")),
        };
        buf = (buf << 6) | u32::from(val);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xFF) as u8);
        }
    }
    Ok(out)
}

#[cfg(target_os = "macos")]
fn spatial_zone(left_x: f32, top_y: f32, width_n: f32) -> String {
    let v = if top_y < 0.24 {
        "top"
    } else if top_y < 0.72 {
        "mid"
    } else {
        "bottom"
    };
    let center_x = left_x + width_n * 0.5;
    let h = if width_n > 0.58 || (0.34..=0.66).contains(&center_x) {
        "center"
    } else if center_x < 0.34 {
        "left"
    } else {
        "right"
    };
    format!("{v}-{h}")
}

#[cfg(target_os = "macos")]
struct PaletteSummary {
    theme: String,
    bg_hex: String,
    accent_hex: String,
    palette: Vec<(String, u32)>,
    band_colors: [String; 3],
}

#[cfg(target_os = "macos")]
fn analyze_rgba_grid(rgba: &[u8], w: usize, h: usize) -> PaletteSummary {
    if w == 0 || h == 0 || rgba.len() < w * h * 4 {
        return PaletteSummary {
            theme: "light".to_string(),
            bg_hex: "#ffffff".to_string(),
            accent_hex: "#0d9488".to_string(),
            palette: Vec::new(),
            band_colors: ["#ffffff".into(), "#ffffff".into(), "#ffffff".into()],
        };
    }

    // Quantize RGB into 4-bit-per-channel 4096-bin histogram while accumulating exact RGB sums per bin.
    let mut counts = vec![0u32; 4096];
    let mut sum_r = vec![0u64; 4096];
    let mut sum_g = vec![0u64; 4096];
    let mut sum_b = vec![0u64; 4096];
    let mut border_counts = vec![0u32; 4096];
    let mut band_sums = [[0u64; 3]; 3];
    let mut band_px = [0u64; 3];
    let mut lum_sum = 0.0f64;
    let total_px = (w * h) as u64;

    for y in 0..h {
        let band = if y * 4 < h {
            0
        } else if y * 4 < h * 3 {
            1
        } else {
            2
        };
        let is_y_border = y < 2 || y + 2 >= h;
        for x in 0..w {
            let idx = (y * w + x) * 4;
            let a = rgba[idx + 3];
            // Composite transparent pixels over white so UI icons preserve expected contrast.
            let (r, g, b) = if a == 255 {
                (u64::from(rgba[idx]), u64::from(rgba[idx + 1]), u64::from(rgba[idx + 2]))
            } else {
                let af = u64::from(a);
                let inv = 255 - af;
                (
                    (u64::from(rgba[idx]) * af + 255 * inv) / 255,
                    (u64::from(rgba[idx + 1]) * af + 255 * inv) / 255,
                    (u64::from(rgba[idx + 2]) * af + 255 * inv) / 255,
                )
            };
            let bin = (((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)) as usize;
            counts[bin] += 1;
            sum_r[bin] += r;
            sum_g[bin] += g;
            sum_b[bin] += b;
            if is_y_border || x < 2 || x + 2 >= w {
                border_counts[bin] += 1;
            }
            band_sums[band][0] += r;
            band_sums[band][1] += g;
            band_sums[band][2] += b;
            band_px[band] += 1;
            lum_sum += (0.2126 * r as f64 + 0.7152 * g as f64 + 0.0722 * b as f64) / 255.0;
        }
    }

    let mean_lum = lum_sum / (total_px.max(1) as f64);
    let theme = if mean_lum < 0.34 {
        "dark"
    } else if mean_lum > 0.68 {
        "light"
    } else {
        "mid-tone"
    }
    .to_string();

    let bin_rgb = |bin: usize| -> (u8, u8, u8) {
        let c = u64::from(counts[bin]).max(1);
        (
            (sum_r[bin] / c) as u8,
            (sum_g[bin] / c) as u8,
            (sum_b[bin] / c) as u8,
        )
    };
    let fmt_hex = |(r, g, b): (u8, u8, u8)| format!("#{r:02x}{g:02x}{b:02x}");

    let bg_bin = (0..4096)
        .max_by_key(|&b| u64::from(border_counts[b]) * 3 + u64::from(counts[b]))
        .unwrap_or(0);
    let bg_rgb = bin_rgb(bg_bin);
    let bg_hex = fmt_hex(bg_rgb);

    // Merge nearby top bins (Manhattan RGB distance < 26) so palette entries are distinct.
    let mut active_bins: Vec<usize> = (0..4096).filter(|&b| counts[b] > 0).collect();
    active_bins.sort_unstable_by(|&a, &b| counts[b].cmp(&counts[a]));

    let mut merged: Vec<((u8, u8, u8), u32)> = Vec::new();
    for bin in active_bins {
        let rgb = bin_rgb(bin);
        let cnt = counts[bin];
        if let Some(existing) = merged.iter_mut().find(|((er, eg, eb), _)| {
            i32::from(*er).abs_diff(i32::from(rgb.0))
                + i32::from(*eg).abs_diff(i32::from(rgb.1))
                + i32::from(*eb).abs_diff(i32::from(rgb.2))
                < 28
        }) {
            existing.1 += cnt;
        } else if merged.len() < 12 {
            merged.push((rgb, cnt));
        }
    }
    merged.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    // Pick accent color maximizing saturation + contrast against `bg_rgb`.
    let mut best_accent = merged.first().map(|x| x.0).unwrap_or(bg_rgb);
    let mut best_score = -1.0f64;
    for &(rgb, cnt) in &merged {
        if u64::from(cnt) * 500 < total_px {
            continue;
        }
        let max_c = rgb.0.max(rgb.1).max(rgb.2) as f64;
        let min_c = rgb.0.min(rgb.1).min(rgb.2) as f64;
        let sat = if max_c > 1.0 { (max_c - min_c) / max_c } else { 0.0 };
        let dist = (f64::from(rgb.0) - f64::from(bg_rgb.0)).abs()
            + (f64::from(rgb.1) - f64::from(bg_rgb.1)).abs()
            + (f64::from(rgb.2) - f64::from(bg_rgb.2)).abs();
        let score = dist * (0.45 + sat * 1.55) * (cnt as f64).powf(0.18);
        if dist > 30.0 && score > best_score {
            best_score = score;
            best_accent = rgb;
        }
    }
    let accent_hex = fmt_hex(best_accent);

    let palette: Vec<(String, u32)> = merged
        .into_iter()
        .take(5)
        .map(|(rgb, cnt)| {
            let pct = ((u64::from(cnt) * 100 + total_px / 2) / total_px.max(1)).max(1) as u32;
            (fmt_hex(rgb), pct)
        })
        .collect();

    let band_hex = |b: usize| -> String {
        let n = band_px[b].max(1);
        fmt_hex((
            (band_sums[b][0] / n) as u8,
            (band_sums[b][1] / n) as u8,
            (band_sums[b][2] / n) as u8,
        ))
    };

    PaletteSummary {
        theme,
        bg_hex,
        accent_hex,
        palette,
        band_colors: [band_hex(0), band_hex(1), band_hex(2)],
    }
}

pub fn analyze_image_payload(
    payload: &str,
    workdir: Option<&Path>,
    slot: usize,
) -> Result<ImagePerception, String> {
    let (ext, bytes) = decode_image_payload(payload)?;
    analyze_image_bytes(&bytes, &ext, workdir, slot)
}

/// Warms the Apple Vision framework pipeline once at startup so first user paste completes in `< 100 ms`.
pub fn prewarm_vision() {
    // Minimal 1x1 transparent PNG.
    const TINY_PNG: &[u8] = &[
        0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, b'I', b'H', b'D',
        b'R', 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0B, b'I', b'D', b'A', b'T', 0x78, 0x01, 0x63, 0x60,
        0x00, 0x02, 0x00, 0x00, 0x05, 0x00, 0x01, 0xE2, 0x26, 0x05, 0x9B, 0x00, 0x00, 0x00, 0x00,
        b'I', b'E', b'N', b'D', 0xAE, 0x42, 0x60, 0x82,
    ];
    let _ = analyze_image_bytes(TINY_PNG, "png", None, 0);
}

pub fn analyze_image_bytes(
    bytes: &[u8],
    ext: &str,
    workdir: Option<&Path>,
    slot: usize,
) -> Result<ImagePerception, String> {
    let t0 = Instant::now();
    let saved_rel_path = if let Some(dir) = workdir {
        let media_dir = dir.join(".kronos_media");
        if std::fs::create_dir_all(&media_dir).is_ok() {
            let fname = format!("image_{slot}.{ext}");
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

    #[cfg(target_os = "macos")]
    {
        macos::analyze_macos(bytes, ext, saved_rel_path, t0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let (w, h) = fallback_dimensions(bytes);
        Ok(ImagePerception {
            width: w,
            height: h,
            format_ext: ext.to_string(),
            theme: "unknown".to_string(),
            bg_hex: "#ffffff".to_string(),
            accent_hex: "#0d9488".to_string(),
            palette: Vec::new(),
            band_colors: ["#ffffff".into(), "#ffffff".into(), "#ffffff".into()],
            scene_tags: Vec::new(),
            ocr_regions: Vec::new(),
            saved_rel_path,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
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

    #[link(name = "Vision", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFDataCreate(allocator: *const c_void, bytes: *const u8, length: isize) -> *const c_void;
        fn CFRelease(cf: *const c_void);
    }

    #[link(name = "ImageIO", kind = "framework")]
    unsafe extern "C" {
        fn CGImageSourceCreateWithData(data: *const c_void, options: *const c_void) -> *const c_void;
        fn CGImageSourceCreateImageAtIndex(
            isrc: *const c_void,
            index: usize,
            options: *const c_void,
        ) -> *const c_void;
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGImageGetWidth(image: *const c_void) -> usize;
        fn CGImageGetHeight(image: *const c_void) -> usize;
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
        fn CGContextDrawImage(c: *const c_void, rect: CGRect, image: *const c_void);
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

    pub(super) fn analyze_macos(
        bytes: &[u8],
        ext: &str,
        saved_rel_path: Option<String>,
        t0: Instant,
    ) -> Result<ImagePerception, String> {
        // SAFETY: CoreGraphics + Vision objects are scoped to this autorelease pool and released before return.
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let res = analyze_macos_inner(bytes, ext, saved_rel_path, t0);
            objc_autoreleasePoolPop(pool);
            res
        }
    }

    unsafe fn analyze_macos_inner(
        bytes: &[u8],
        ext: &str,
        saved_rel_path: Option<String>,
        t0: Instant,
    ) -> Result<ImagePerception, String> {
        let cf_data = unsafe { CFDataCreate(std::ptr::null(), bytes.as_ptr(), bytes.len() as isize) };
        if cf_data.is_null() {
            return Err("CFDataCreate failed".to_string());
        }
        let img_src = unsafe { CGImageSourceCreateWithData(cf_data, std::ptr::null()) };
        unsafe { CFRelease(cf_data) };
        if img_src.is_null() {
            return Err("unsupported or corrupt image container".to_string());
        }
        let cg_img = unsafe { CGImageSourceCreateImageAtIndex(img_src, 0, std::ptr::null()) };
        unsafe { CFRelease(img_src) };
        if cg_img.is_null() {
            return Err("unable to decode image frame".to_string());
        }

        let width = unsafe { CGImageGetWidth(cg_img) };
        let height = unsafe { CGImageGetHeight(cg_img) };
        if width == 0 || height == 0 {
            unsafe { CGImageRelease(cg_img) };
            return Err("decoded image has zero dimensions".to_string());
        }

        // Downsample to <= 320 px along longest edge for sub-millisecond palette, bubble tint, and spatial band analysis.
        let scale = (320.0 / (width.max(height) as f64)).min(1.0);
        let sw = ((width as f64 * scale).round() as usize).max(1);
        let sh = ((height as f64 * scale).round() as usize).max(1);
        let mut rgba = vec![0u8; sw * sh * 4];
        let cs = unsafe { CGColorSpaceCreateDeviceRGB() };
        // kCGImageAlphaPremultipliedLast (1) | kCGBitmapByteOrder32Big (0x4000) -> RGBA8 bytes.
        let ctx = unsafe {
            CGBitmapContextCreate(rgba.as_mut_ptr(), sw, sh, 8, sw * 4, cs, 0x4001)
        };
        if !ctx.is_null() {
            let rect = CGRect {
                origin: CGPoint { x: 0.0, y: 0.0 },
                size: CGSize {
                    width: sw as f64,
                    height: sh as f64,
                },
            };
            unsafe {
                CGContextDrawImage(ctx, rect, cg_img);
                CGContextRelease(ctx);
            }
        }
        unsafe { CGColorSpaceRelease(cs) };
        let pal = analyze_rgba_grid(&rgba, sw, sh);

        let ocr_regions = unsafe { run_vision_ocr(cg_img, &rgba, sw, sh) };
        let scene_tags = unsafe { run_vision_classify(cg_img) };
        unsafe { CGImageRelease(cg_img) };

        Ok(ImagePerception {
            width,
            height,
            format_ext: ext.to_string(),
            theme: pal.theme,
            bg_hex: pal.bg_hex,
            accent_hex: pal.accent_hex,
            palette: pal.palette,
            band_colors: pal.band_colors,
            scene_tags,
            ocr_regions,
            saved_rel_path,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn sample_region_visual_cues(
        rgba: &[u8],
        sw: usize,
        sh: usize,
        left_x: f32,
        top_y: f32,
        width_n: f32,
        height_n: f32,
    ) -> ([u8; 3], bool, bool) {
        if sw == 0 || sh == 0 || rgba.len() < sw * sh * 4 {
            return ([0, 0, 0], false, false);
        }
        let x0 = ((left_x * sw as f32).floor() as isize - 1).clamp(0, sw as isize - 1) as usize;
        let x1 = (((left_x + width_n) * sw as f32).ceil() as isize + 1)
            .clamp(x0 as isize, sw as isize - 1) as usize;
        let y0 = ((top_y * sh as f32).floor() as isize - 1).clamp(0, sh as isize - 1) as usize;
        let y1 = (((top_y + height_n) * sh as f32).ceil() as isize + 1)
            .clamp(y0 as isize, sh as isize - 1) as usize;

        let mut total_px = 0u32;
        let mut green_px = 0u32;
        let mut green_sum = [0u64; 3];
        let mut blue_px = 0u32;
        let mut blue_sum = [0u64; 3];
        let mut mid_px = 0u32;
        let mut mid_sum = [0u64; 3];
        let mut all_sum = [0u64; 3];

        for y in y0..=y1 {
            for x in x0..=x1 {
                let idx = (y * sw + x) * 4;
                let r = i32::from(rgba[idx]);
                let g = i32::from(rgba[idx + 1]);
                let b = i32::from(rgba[idx + 2]);
                total_px += 1;
                all_sum[0] += r as u64;
                all_sum[1] += g as u64;
                all_sum[2] += b as u64;

                if g - r >= 15 && g - b >= 7 {
                    green_px += 1;
                    green_sum[0] += r as u64;
                    green_sum[1] += g as u64;
                    green_sum[2] += b as u64;
                } else if b - r >= 24 && b - g >= 14 {
                    blue_px += 1;
                    blue_sum[0] += r as u64;
                    blue_sum[1] += g as u64;
                    blue_sum[2] += b as u64;
                }
                let max_c = r.max(g).max(b);
                if (16..=205).contains(&max_c) {
                    mid_px += 1;
                    mid_sum[0] += r as u64;
                    mid_sum[1] += g as u64;
                    mid_sum[2] += b as u64;
                }
            }
        }

        let bg_rgb = if total_px > 0 && green_px * 100 / total_px >= 18 {
            [
                (green_sum[0] / u64::from(green_px)) as u8,
                (green_sum[1] / u64::from(green_px)) as u8,
                (green_sum[2] / u64::from(green_px)) as u8,
            ]
        } else if total_px > 0 && blue_px * 100 / total_px >= 18 {
            [
                (blue_sum[0] / u64::from(blue_px)) as u8,
                (blue_sum[1] / u64::from(blue_px)) as u8,
                (blue_sum[2] / u64::from(blue_px)) as u8,
            ]
        } else if mid_px > 0 {
            [
                (mid_sum[0] / u64::from(mid_px)) as u8,
                (mid_sum[1] / u64::from(mid_px)) as u8,
                (mid_sum[2] / u64::from(mid_px)) as u8,
            ]
        } else if total_px > 0 {
            [
                (all_sum[0] / u64::from(total_px)) as u8,
                (all_sum[1] / u64::from(total_px)) as u8,
                (all_sum[2] / u64::from(total_px)) as u8,
            ]
        } else {
            [0, 0, 0]
        };

        // Check narrow strip immediately left of the OCR span for:
        // 1. High-saturation vertical reply-quote accent bar (pink/purple/teal/orange inside message bubbles).
        // 2. Sidebar outgoing read/delivery checkmark glyph pixels (cyan/blue #53bdeb or gray tick stroke).
        let lx0 = (((left_x - 0.026).max(0.0)) * sw as f32).floor() as usize;
        let lx1 = (((left_x - 0.002).max(0.0)) * sw as f32).ceil() as usize;
        let mut quote_bar_hits = 0u32;
        let mut blue_tick_hits = 0u32;
        let mut gray_tick_hits = 0u32;
        if lx0 <= lx1 && lx1 < sw {
            let bg_lum = i32::from(bg_rgb[0]) + i32::from(bg_rgb[1]) + i32::from(bg_rgb[2]);
            for y in y0..=y1 {
                for x in lx0..=lx1 {
                    let idx = (y * sw + x) * 4;
                    let r = i32::from(rgba[idx]);
                    let g = i32::from(rgba[idx + 1]);
                    let b = i32::from(rgba[idx + 2]);
                    let max_c = r.max(g).max(b);
                    let min_c = r.min(g).min(b);
                    let sat = max_c - min_c;
                    if max_c >= 115 && sat >= 46 {
                        quote_bar_hits += 1;
                    }
                    if b >= 140 && g >= 110 && (b - r) >= 35 {
                        blue_tick_hits += 1;
                    }
                    let px_lum = r + g + b;
                    if (px_lum - bg_lum).abs() >= 95 && sat <= 35 {
                        gray_tick_hits += 1;
                    }
                }
            }
        }
        let span_rows = (y1.saturating_sub(y0) + 1) as u32;
        let has_left_quote_bar = quote_bar_hits >= span_rows.div_ceil(2).max(2);
        let has_left_tick_icon = blue_tick_hits >= 1 || gray_tick_hits >= 2;
        (bg_rgb, has_left_quote_bar, has_left_tick_icon)
    }

    unsafe fn run_vision_ocr(
        cg_img: *const c_void,
        rgba: &[u8],
        sw: usize,
        sh: usize,
    ) -> Vec<OcrRegion> {
        let mut regions = Vec::new();
        let handler_cls = class("VNImageRequestHandler");
        let req_cls = class("VNRecognizeTextRequest");
        if handler_cls.is_null() || req_cls.is_null() {
            return regions;
        }
        unsafe {
            let empty_dict: Id = msg!(class("NSDictionary"), "dictionary"; Id);
            let handler_alloc: Id = msg!(handler_cls, "alloc"; Id);
            let handler: Id = msg!(
                handler_alloc,
                "initWithCGImage:options:",
                cg_img => *const c_void,
                empty_dict => Id;
                Id
            );
            if handler.is_null() {
                return regions;
            }
            let req_alloc: Id = msg!(req_cls, "alloc"; Id);
            let req: Id = msg!(req_alloc, "init"; Id);
            if !req.is_null() {
                // 0 = VNRequestTextRecognitionLevelAccurate.
                let _: () = msg!(req, "setRecognitionLevel:", 0isize => isize; ());
                let _: () = msg!(req, "setUsesLanguageCorrection:", true => bool; ());
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
                            let text = CStr::from_ptr(c_ptr).to_string_lossy().trim().to_string();
                            if text.is_empty() {
                                continue;
                            }
                            let top_y = (1.0 - (bbox.origin.y + bbox.size.height)).clamp(0.0, 1.0) as f32;
                            let left_x = bbox.origin.x.clamp(0.0, 1.0) as f32;
                            let width_n = bbox.size.width.clamp(0.0, 1.0) as f32;
                            let height_n = bbox.size.height.clamp(0.0, 1.0) as f32;
                            let (bg_rgb, has_left_quote_bar, has_left_tick_icon) =
                                sample_region_visual_cues(
                                    rgba, sw, sh, left_x, top_y, width_n, height_n,
                                );
                            regions.push(OcrRegion {
                                zone: spatial_zone(left_x, top_y, width_n),
                                text,
                                top_y,
                                left_x,
                                width_n,
                                height_n,
                                bg_rgb,
                                has_left_quote_bar,
                                has_left_tick_icon,
                            });
                        }
                    }
                }
                let _: () = msg!(req, "release"; ());
            }
            let _: () = msg!(handler, "release"; ());
        }
        regions.sort_by(|a, b| {
            let row_a = (a.top_y * 25.0).round() as i32;
            let row_b = (b.top_y * 25.0).round() as i32;
            row_a
                .cmp(&row_b)
                .then_with(|| a.left_x.partial_cmp(&b.left_x).unwrap_or(std::cmp::Ordering::Equal))
        });
        regions
    }

    unsafe fn run_vision_classify(cg_img: *const c_void) -> Vec<(String, u32)> {
        let mut tags = Vec::new();
        let handler_cls = class("VNImageRequestHandler");
        let req_cls = class("VNClassifyImageRequest");
        if handler_cls.is_null() || req_cls.is_null() {
            return tags;
        }
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
                return tags;
            }
            let req: Id = msg!(msg!(req_cls, "alloc"; Id), "init"; Id);
            if !req.is_null() {
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
                        for i in 0..n.min(24) {
                            let obs: Id = msg!(results, "objectAtIndex:", i => usize; Id);
                            if obs.is_null() {
                                continue;
                            }
                            let conf: f32 = msg!(obs, "confidence"; f32);
                            if conf < 0.18 {
                                continue;
                            }
                            let ident: Id = msg!(obs, "identifier"; Id);
                            if ident.is_null() {
                                continue;
                            }
                            let c_ptr: *const c_char = msg!(ident, "UTF8String"; *const c_char);
                            if c_ptr.is_null() {
                                continue;
                            }
                            let label = CStr::from_ptr(c_ptr).to_string_lossy().replace('_', " ");
                            tags.push((label, (conf * 100.0).round() as u32));
                            if tags.len() >= 5 {
                                break;
                            }
                        }
                    }
                }
                let _: () = msg!(req, "release"; ());
            }
            let _: () = msg!(handler, "release"; ());
        }
        tags
    }
}

#[cfg(not(target_os = "macos"))]
fn fallback_dimensions(bytes: &[u8]) -> (usize, usize) {
    if bytes.len() >= 24 && bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
        let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]) as usize;
        return (w, h);
    }
    (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crc32_png(chunk_type: &[u8; 4], data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in chunk_type.iter().chain(data.iter()) {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    fn adler32(data: &[u8]) -> u32 {
        let mut s1 = 1u32;
        let mut s2 = 0u32;
        for &b in data {
            s1 = (s1 + u32::from(b)) % 65521;
            s2 = (s2 + s1) % 65521;
        }
        (s2 << 16) | s1
    }

    fn write_png_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(tag);
        out.extend_from_slice(data);
        out.extend_from_slice(&crc32_png(tag, data).to_be_bytes());
    }

    fn encode_uncompressed_rgba_png(w: usize, h: usize, rgba: &[u8]) -> Vec<u8> {
        let mut raw = Vec::with_capacity(h * (1 + w * 4));
        for y in 0..h {
            raw.push(0); // filter type None
            raw.extend_from_slice(&rgba[y * w * 4..(y + 1) * w * 4]);
        }
        let mut z = vec![0x78, 0x01]; // zlib header
        let chunks: Vec<&[u8]> = raw.chunks(65535).collect();
        for (i, ch) in chunks.iter().enumerate() {
            let is_last = u8::from(i + 1 == chunks.len());
            z.push(is_last);
            let len = ch.len() as u16;
            z.extend_from_slice(&len.to_le_bytes());
            z.extend_from_slice(&(!len).to_le_bytes());
            z.extend_from_slice(ch);
        }
        z.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&(w as u32).to_be_bytes());
        ihdr.extend_from_slice(&(h as u32).to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA
        write_png_chunk(&mut png, b"IHDR", &ihdr);
        write_png_chunk(&mut png, b"IDAT", &z);
        write_png_chunk(&mut png, b"IEND", &[]);
        png
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

    #[test]
    fn test_data_url_decode_palette_and_macos_vision() {
        let (w, h) = (240usize, 120usize);
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                // Dark slate background (#0f172a) with teal button accent (#14b8a6).
                if (60..180).contains(&x) && (40..80).contains(&y) {
                    rgba[idx..idx + 4].copy_from_slice(&[0x14, 0xb8, 0xa6, 0xff]);
                } else {
                    rgba[idx..idx + 4].copy_from_slice(&[0x0f, 0x17, 0x2a, 0xff]);
                }
            }
        }
        let png = encode_uncompressed_rgba_png(w, h, &rgba);
        let data_url = format!("data:image/png;base64,{}", encode_base64(&png));
        let tmp = std::env::temp_dir().join(format!("kronos_vision_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let p = analyze_image_payload(&data_url, Some(&tmp), 1).expect("image analysis failed");
        assert_eq!(p.width, 240);
        assert_eq!(p.height, 120);
        assert_eq!(p.theme, "dark");
        assert_eq!(p.bg_hex, "#0f172a");
        assert_eq!(p.accent_hex, "#14b8a6");
        assert_eq!(p.saved_rel_path.as_deref(), Some(".kronos_media/image_1.png"));
        assert!(tmp.join(".kronos_media/image_1.png").is_file());
        let ctx = p.to_prompt_context(1);
        assert!(ctx.contains("240x120 PNG"));
        assert!(ctx.contains("#0f172a"));
        assert!(ctx.contains("#14b8a6"));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
