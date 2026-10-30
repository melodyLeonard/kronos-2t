//! Zero-crate C FFI surface for iOS (`aarch64-apple-ios`) and Swift host applications.
//!
//! Designed for apps linked against `libkronos_2t.a` with the
//! `com.apple.developer.kernel.increased-memory-limit` entitlement (`Mobile5G` `<= 4.9 GiB`
//! working-set envelope). Every `extern "C"` entrypoint traps Rust panics before the FFI boundary.

use crate::runtime::chat::{ChatModel, GenParams, Message};
use crate::runtime::jev::CalibratedChoice;
use crate::util::json::Json;
use crate::model::memory_budget::{current_resident_bytes, MemoryBudgetProfile};
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

/// Streaming callback receiving `(utf8_ptr, utf8_len, user_data)` and returning `true` to continue.
pub type KronosStreamCallback =
    Option<unsafe extern "C" fn(*const c_char, usize, *mut c_void) -> bool>;

/// Opaque session handle owned across the C / Swift FFI boundary.
pub struct KronosMobileSession {
    model: ChatModel,
    pending_token: Option<usize>,
    raw_bytes: Vec<u8>,
    emitted_bytes: usize,
    last_error: CString,
}

impl KronosMobileSession {
    fn set_error(&mut self, msg: impl Into<String>) {
        let cleaned = msg.into().replace('\0', " ");
        self.last_error =
            CString::new(cleaned).unwrap_or_else(|_| CString::new("ffi error").unwrap());
    }

    fn clear_error(&mut self) {
        self.last_error = CString::new("").unwrap();
    }
}

unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err("null C string pointer".to_string());
    }
    // SAFETY: Caller guarantees `ptr` points to a valid NUL-terminated C string for the call duration.
    let cs = unsafe { CStr::from_ptr(ptr) };
    cs.to_str().map_err(|e| format!("invalid UTF-8 string: {e}"))
}

unsafe fn opt_cstr_to_str<'a>(ptr: *const c_char) -> Result<Option<&'a str>, String> {
    if ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: Non-null pointer forwarded to `cstr_to_str`.
    let s = unsafe { cstr_to_str(ptr)? };
    if s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(s))
    }
}

/// Creates a new mobile inference session from `model_dir_utf8` (`kronos-<mode>.k2tb` or
/// `kronos-moe-core-<mode>.k2tb` + `.k2tm`).
///
/// When `memory_budget_mb > 0`, installs the process-wide `MemoryBudgetProfile` cap before loading
/// so SLRU expert arenas, `Q4` KV cache defaults, and `256`-token Metal prefill chunks match the
/// device envelope (e.g. `4900` on 8 GB iOS devices, `5120` on 8 GB MacBooks).
///
/// # Safety
/// `model_dir_utf8` and `mode_utf8` must be valid NUL-terminated UTF-8 C strings. The returned
/// pointer must be released with [`kronos_session_destroy`].
#[no_mangle]
pub unsafe extern "C" fn kronos_session_create(
    model_dir_utf8: *const c_char,
    mode_utf8: *const c_char,
    ctx_tokens: usize,
    threads: usize,
    memory_budget_mb: usize,
) -> *mut KronosMobileSession {
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<KronosMobileSession, String> {
        // SAFETY: Validated non-null NUL-terminated pointers per FFI contract.
        let model_dir = unsafe { cstr_to_str(model_dir_utf8)? };
        // SAFETY: Validated non-null NUL-terminated pointers per FFI contract.
        let mode = unsafe { cstr_to_str(mode_utf8)? };
        if memory_budget_mb > 0 {
            MemoryBudgetProfile::set_global_budget_override_mb(memory_budget_mb);
        }
        let ctx = ctx_tokens.max(64);
        let n_threads = if threads == 0 {
            crate::kernels::thread_pool::default_inference_threads()
        } else {
            threads
        };
        let calib_candidate = Path::new(model_dir).join("calib.txt");
        let calib = calib_candidate
            .exists()
            .then_some((calib_candidate.as_path(), 512usize));
        let model = ChatModel::load(Path::new(model_dir), mode, ctx, n_threads, calib)?;
        Ok(KronosMobileSession {
            model,
            pending_token: None,
            raw_bytes: Vec::new(),
            emitted_bytes: 0,
            last_error: CString::new("").unwrap(),
        })
    }));

    match res {
        Ok(Ok(sess)) => Box::into_raw(Box::new(sess)),
        _ => std::ptr::null_mut(),
    }
}

/// Releases a `KronosMobileSession` created by [`kronos_session_create`].
///
/// # Safety
/// `session` must be either null or a unique live pointer returned by [`kronos_session_create`].
#[no_mangle]
pub unsafe extern "C" fn kronos_session_destroy(session: *mut KronosMobileSession) {
    if session.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Exclusive ownership transferred back from `Box::into_raw`.
        unsafe {
            drop(Box::from_raw(session));
        }
    }));
}

/// Returns the last error string recorded on `session` (valid until the next mutating call on
/// `session`), or an empty static string when `session` is null.
///
/// # Safety
/// `session` must be null or a valid `KronosMobileSession` pointer.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_last_error(
    session: *const KronosMobileSession,
) -> *const c_char {
    if session.is_null() {
        return c"null session".as_ptr();
    }
    // SAFETY: Caller guarantees `session` is a live pointer with no concurrent mutation.
    let sess = unsafe { &*session };
    sess.last_error.as_ptr()
}

/// Enables Apple Metal GPU execution on the target model (`0` on success, `-1` on error).
///
/// # Safety
/// `session` must be a valid `KronosMobileSession` pointer.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_enable_metal(session: *mut KronosMobileSession) -> i32 {
    if session.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Validated non-null exclusive session pointer.
        let sess = unsafe { &mut *session };
        match sess.model.enable_metal() {
            Ok(()) => {
                sess.pending_token = None;
                sess.clear_error();
                0
            }
            Err(e) => {
                sess.set_error(e);
                -1
            }
        }
    }));
    res.unwrap_or(-1)
}

/// Configures KV-cache bit-width (`4`, `8`, or `32`) and optional sliding-window + attention-sink
/// retention (`sliding_window == 0` disables windowing). Returns `0` on success, `-1` on error.
///
/// # Safety
/// `session` must be a valid `KronosMobileSession` pointer.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_configure_kv(
    session: *mut KronosMobileSession,
    kv_bits: u8,
    sliding_window: usize,
    sink_tokens: usize,
) -> i32 {
    if session.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Validated non-null exclusive session pointer.
        let sess = unsafe { &mut *session };
        if !matches!(kv_bits, 4 | 8 | 32) {
            sess.set_error(format!("unsupported kv_bits={kv_bits}; expected 4, 8, or 32"));
            return -1;
        }
        sess.model.set_kv_bits(kv_bits);
        let win = (sliding_window > 0).then_some(sliding_window);
        sess.model.set_kv_sliding_window(win, sink_tokens);
        sess.pending_token = None;
        sess.clear_error();
        0
    }));
    res.unwrap_or(-1)
}

/// Attaches a small CPU draft model for speculative decoding and `Kronos-Jev` anchor contrast.
/// Returns `0` on success, `-1` on error.
///
/// # Safety
/// `session`, `draft_dir_utf8`, and `draft_mode_utf8` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_attach_draft(
    session: *mut KronosMobileSession,
    draft_dir_utf8: *const c_char,
    draft_mode_utf8: *const c_char,
) -> i32 {
    if session.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Validated non-null exclusive session pointer.
        let sess = unsafe { &mut *session };
        // SAFETY: Validated C string pointers per FFI contract.
        let draft_dir = match unsafe { cstr_to_str(draft_dir_utf8) } {
            Ok(s) => s,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        // SAFETY: Validated C string pointers per FFI contract.
        let draft_mode = match unsafe { cstr_to_str(draft_mode_utf8) } {
            Ok(s) => s,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        match sess.model.attach_draft(Path::new(draft_dir), draft_mode, None) {
            Ok(()) => {
                sess.clear_error();
                0
            }
            Err(e) => {
                sess.set_error(e);
                -1
            }
        }
    }));
    res.unwrap_or(-1)
}

/// Executes a single-forward-pass `Kronos-Jev` calibrated decision (`0` decode tokens) over a JSON
/// array of candidate anchor labels (`options_json_utf8`, e.g. `["Chat","Computer","Agent"]` or
/// `["Yes","No"]`), preserving the session's active KV cache.
///
/// Writes calibrated probabilities into `out_probs_f32` (up to `max_options`), optional Shannon
/// entropy into `out_entropy_f32`, and normalized confidence `[0, 1]` into `out_confidence_f32`.
/// Returns the selected `0`-based option index (`>= 0`), or `-1` on error.
///
/// # Safety
/// `session`, `query_utf8`, and `options_json_utf8` must be valid non-null pointers. Output
/// pointers may be null when unused; if `out_probs_f32` is non-null it must have capacity for at
/// least `max_options` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_jev_decide(
    session: *mut KronosMobileSession,
    system_utf8: *const c_char,
    query_utf8: *const c_char,
    options_json_utf8: *const c_char,
    temperature: f32,
    out_probs_f32: *mut f32,
    max_options: usize,
    out_entropy_f32: *mut f32,
    out_confidence_f32: *mut f32,
) -> i32 {
    if session.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Validated non-null exclusive session pointer.
        let sess = unsafe { &mut *session };
        // SAFETY: Optional C string pointer checked for null.
        let sys_opt = match unsafe { opt_cstr_to_str(system_utf8) } {
            Ok(v) => v,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        // SAFETY: Validated C string pointer per FFI contract.
        let query = match unsafe { cstr_to_str(query_utf8) } {
            Ok(v) => v,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        // SAFETY: Validated C string pointer per FFI contract.
        let opts_raw = match unsafe { cstr_to_str(options_json_utf8) } {
            Ok(v) => v,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        let parsed = match Json::parse(opts_raw) {
            Ok(Json::Arr(items)) => items,
            Ok(_) => {
                sess.set_error("options_json_utf8 must be a JSON array of strings");
                return -1;
            }
            Err(e) => {
                sess.set_error(format!("invalid options_json_utf8: {e}"));
                return -1;
            }
        };
        let mut owned_opts: Vec<String> = Vec::with_capacity(parsed.len());
        for item in &parsed {
            if let Some(s) = item.as_str() {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    owned_opts.push(trimmed.to_string());
                }
            }
        }
        if owned_opts.len() < 2 {
            sess.set_error("options_json_utf8 requires at least 2 non-empty labels");
            return -1;
        }
        let anchor_refs: Vec<&str> = owned_opts.iter().map(String::as_str).collect();
        let default_sys =
            "Select the single most accurate option label and respond with only that label.";
        let sys_prompt = sys_opt.unwrap_or(default_sys);
        let t0 = std::time::Instant::now();
        let Some((anchor_logprobs, _mass)) =
            sess.model.jev_score_anchors(sys_prompt, query, &anchor_refs)
        else {
            sess.set_error("jev_score_anchors could not find distinct single-token anchors");
            return -1;
        };
        let temp = if temperature > 0.0 { temperature } else { 1.0 };
        let choice = CalibratedChoice::from_logits(
            &anchor_logprobs,
            temp,
            true,
            t0.elapsed().as_secs_f64() * 1000.0,
        );
        if !out_probs_f32.is_null() && max_options > 0 {
            let n = choice.probs.len().min(max_options);
            // SAFETY: Caller guarantees `out_probs_f32` points to at least `max_options` `f32`s.
            let dst = unsafe { std::slice::from_raw_parts_mut(out_probs_f32, n) };
            dst.copy_from_slice(&choice.probs[..n]);
        }
        if !out_entropy_f32.is_null() {
            // SAFETY: Non-null aligned output `f32` pointer provided by caller.
            unsafe {
                *out_entropy_f32 = choice.entropy;
            }
        }
        if !out_confidence_f32.is_null() {
            // SAFETY: Non-null aligned output `f32` pointer provided by caller.
            unsafe {
                *out_confidence_f32 = choice.confidence;
            }
        }
        sess.clear_error();
        choice.selected as i32
    }));
    res.unwrap_or(-1)
}

/// Prefills a ChatML turn (`[system, user]`, reusing any cached KV prefix) and stages the first
/// predicted token for [`kronos_session_step`]. Returns the initial token ID (`>= 0`), or `-1` on
/// error.
///
/// # Safety
/// `session` and `user_utf8` must be valid non-null pointers; `system_utf8` may be null to use the
/// model's default system prompt.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_prefill(
    session: *mut KronosMobileSession,
    system_utf8: *const c_char,
    user_utf8: *const c_char,
) -> i32 {
    if session.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Validated non-null exclusive session pointer.
        let sess = unsafe { &mut *session };
        // SAFETY: Optional C string pointer checked for null.
        let sys_opt = match unsafe { opt_cstr_to_str(system_utf8) } {
            Ok(v) => v,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        // SAFETY: Validated non-null C string pointer.
        let user = match unsafe { cstr_to_str(user_utf8) } {
            Ok(v) => v,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        let mut msgs = Vec::with_capacity(2);
        if let Some(sys) = sys_opt {
            msgs.push(Message::new("system", sys));
        }
        msgs.push(Message::new("user", user));
        match sess.model.prefill_messages(&msgs) {
            Ok((first_tok, _prefilled)) => {
                sess.pending_token = Some(first_tok);
                sess.raw_bytes.clear();
                sess.emitted_bytes = 0;
                sess.clear_error();
                first_tok as i32
            }
            Err(e) => {
                sess.pending_token = None;
                sess.set_error(e);
                -1
            }
        }
    }));
    res.unwrap_or(-1)
}

/// Emits one greedy decode step after [`kronos_session_prefill`].
///
/// Returns:
/// - `1` when a non-stop token was emitted (writing any newly completed UTF-8 bytes as a
///   NUL-terminated string into `out_utf8_buf` and the emitted token ID into `out_token_id`),
/// - `0` when an end-of-turn / EOS stop token is reached,
/// - `-1` on error (e.g. buffer too small or context limit reached).
///
/// # Safety
/// `session` must be a valid pointer. If `out_utf8_buf` is non-null, it must point to at least
/// `out_buf_cap` writable bytes (`>= 1`). `out_token_id` may be null.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_step(
    session: *mut KronosMobileSession,
    out_utf8_buf: *mut c_char,
    out_buf_cap: usize,
    out_token_id: *mut i32,
) -> i32 {
    if session.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Validated non-null exclusive session pointer.
        let sess = unsafe { &mut *session };
        if !out_utf8_buf.is_null() && out_buf_cap > 0 {
            // SAFETY: At least 1 byte writable at `out_utf8_buf`.
            unsafe {
                *out_utf8_buf = 0;
            }
        }
        let Some(cur_tok) = sess.pending_token else {
            return 0;
        };
        if sess.model.is_stop_token(cur_tok) {
            sess.pending_token = None;
            if !out_token_id.is_null() {
                // SAFETY: Validated non-null output `i32` pointer.
                unsafe {
                    *out_token_id = cur_tok as i32;
                }
            }
            return 0;
        }

        sess.model.append_token_bytes(cur_tok, &mut sess.raw_bytes);
        let pending = &sess.raw_bytes[sess.emitted_bytes..];
        let valid_len = match std::str::from_utf8(pending) {
            Ok(s) => s.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => pending.len(),
        };
        if valid_len > 0 && !out_utf8_buf.is_null() {
            if valid_len + 1 > out_buf_cap {
                sess.set_error(format!(
                    "out_utf8_buf capacity {out_buf_cap} too small for {}-byte piece",
                    valid_len + 1
                ));
                return -1;
            }
            let piece_str = String::from_utf8_lossy(&pending[..valid_len]);
            let piece_bytes = piece_str.as_bytes();
            if piece_bytes.len() + 1 > out_buf_cap {
                sess.set_error("out_utf8_buf capacity exceeded");
                return -1;
            }
            // SAFETY: Bounds checked (`piece_bytes.len() + 1 <= out_buf_cap`).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    piece_bytes.as_ptr(),
                    out_utf8_buf.cast::<u8>(),
                    piece_bytes.len(),
                );
                *out_utf8_buf.add(piece_bytes.len()) = 0;
            }
        }
        sess.emitted_bytes += valid_len;
        if !out_token_id.is_null() {
            // SAFETY: Validated non-null output `i32` pointer.
            unsafe {
                *out_token_id = cur_tok as i32;
            }
        }

        match sess.model.advance_greedy_token(cur_tok) {
            Ok(next_tok) => {
                sess.pending_token = Some(next_tok);
                sess.clear_error();
                1
            }
            Err(e) => {
                sess.pending_token = None;
                sess.set_error(e);
                -1
            }
        }
    }));
    res.unwrap_or(-1)
}

/// Runs full streaming generation (including speculative batch verification when a draft engine is
/// attached), invoking `on_piece_cb` for each UTF-8 text chunk. Returns total generated tokens
/// (`>= 0`), or `-1` on error.
///
/// # Safety
/// `session` and `user_utf8` must be valid pointers; `system_utf8`, `on_piece_cb`, and `user_data`
/// may be null.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_generate(
    session: *mut KronosMobileSession,
    system_utf8: *const c_char,
    user_utf8: *const c_char,
    max_tokens: usize,
    temperature: f32,
    on_piece_cb: KronosStreamCallback,
    user_data: *mut c_void,
) -> i32 {
    if session.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Validated non-null exclusive session pointer.
        let sess = unsafe { &mut *session };
        // SAFETY: Optional C string pointer checked for null.
        let sys_opt = match unsafe { opt_cstr_to_str(system_utf8) } {
            Ok(v) => v,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        // SAFETY: Validated non-null C string pointer.
        let user = match unsafe { cstr_to_str(user_utf8) } {
            Ok(v) => v,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        let mut msgs = Vec::with_capacity(2);
        if let Some(sys) = sys_opt {
            msgs.push(Message::new("system", sys));
        }
        msgs.push(Message::new("user", user));
        let params = GenParams {
            temperature: temperature.max(0.0),
            max_tokens: max_tokens.max(1),
            ..GenParams::default()
        };
        sess.pending_token = None;
        let gen = sess.model.generate(&msgs, &params, |piece| {
            let Some(cb) = on_piece_cb else {
                return true;
            };
            let Ok(cs) = CString::new(piece.replace('\0', "")) else {
                return true;
            };
            // SAFETY: `cs` is a live NUL-terminated buffer for the duration of the callback.
            unsafe { cb(cs.as_ptr(), cs.as_bytes().len(), user_data) }
        });
        match gen {
            Ok(comp) => {
                sess.clear_error();
                comp.completion_tokens as i32
            }
            Err(e) => {
                sess.set_error(e);
                -1
            }
        }
    }));
    res.unwrap_or(-1)
}

/// Disables Apple Metal GPU execution and restores ARM64 CPU execution (`0` on success, `-1` on error).
///
/// # Safety
/// `session` must be a valid `KronosMobileSession` pointer.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_disable_metal(session: *mut KronosMobileSession) -> i32 {
    if session.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Validated non-null exclusive session pointer.
        let sess = unsafe { &mut *session };
        sess.model.disable_metal();
        sess.pending_token = None;
        sess.clear_error();
        0
    }));
    res.unwrap_or(-1)
}

/// Runs multi-turn streaming generation from a JSON array of `{"role":"...","content":"..."}`
/// messages, reusing the session's cached KV prefix across turns and invoking `on_piece_cb` for
/// each streamed UTF-8 piece (returning `false` from `on_piece_cb` cancels generation cleanly).
///
/// Optionally writes `prefill_ms`, `decode_ms`, and `prefilled_tokens` when the output pointers are
/// non-null. Returns total generated tokens (`>= 0`), or `-1` on error.
///
/// # Safety
/// `session` and `messages_json_utf8` must be valid non-null pointers. Output metric pointers,
/// `on_piece_cb`, and `user_data` may be null.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_generate_chat_json(
    session: *mut KronosMobileSession,
    messages_json_utf8: *const c_char,
    max_tokens: usize,
    temperature: f32,
    on_piece_cb: KronosStreamCallback,
    user_data: *mut c_void,
    out_prefill_ms: *mut f64,
    out_decode_ms: *mut f64,
    out_prefilled_tokens: *mut usize,
) -> i32 {
    if session.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Validated non-null exclusive session pointer.
        let sess = unsafe { &mut *session };
        // SAFETY: Validated non-null C string pointer per FFI contract.
        let raw_json = match unsafe { cstr_to_str(messages_json_utf8) } {
            Ok(v) => v,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        let items = match Json::parse(raw_json) {
            Ok(Json::Arr(arr)) => arr,
            Ok(_) => {
                sess.set_error("messages_json_utf8 must be a JSON array of message objects");
                return -1;
            }
            Err(e) => {
                sess.set_error(format!("invalid messages_json_utf8: {e}"));
                return -1;
            }
        };
        let mut msgs = Vec::with_capacity(items.len());
        for item in &items {
            let role = item
                .get("role")
                .and_then(Json::as_str)
                .unwrap_or("")
                .trim();
            let content = item.get("content").and_then(Json::as_str).unwrap_or("");
            if !role.is_empty() && !content.is_empty() {
                msgs.push(Message::new(role, content));
            }
        }
        if msgs.is_empty() {
            sess.set_error("messages_json_utf8 contained zero non-empty messages");
            return -1;
        }
        let params = GenParams {
            temperature: temperature.max(0.0),
            max_tokens: max_tokens.max(1),
            ..GenParams::default()
        };
        sess.pending_token = None;
        let gen = sess.model.generate(&msgs, &params, |piece| {
            let Some(cb) = on_piece_cb else {
                return true;
            };
            let Ok(cs) = CString::new(piece.replace('\0', "")) else {
                return true;
            };
            // SAFETY: `cs` is a live NUL-terminated buffer for the duration of the callback.
            unsafe { cb(cs.as_ptr(), cs.as_bytes().len(), user_data) }
        });
        match gen {
            Ok(comp) => {
                if !out_prefill_ms.is_null() {
                    // SAFETY: Non-null aligned output `f64` pointer provided by caller.
                    unsafe {
                        *out_prefill_ms = comp.prefill_secs * 1000.0;
                    }
                }
                if !out_decode_ms.is_null() {
                    // SAFETY: Non-null aligned output `f64` pointer provided by caller.
                    unsafe {
                        *out_decode_ms = comp.decode_secs * 1000.0;
                    }
                }
                if !out_prefilled_tokens.is_null() {
                    // SAFETY: Non-null aligned output `usize` pointer provided by caller.
                    unsafe {
                        *out_prefilled_tokens = comp.prefilled_tokens;
                    }
                }
                sess.clear_error();
                comp.completion_tokens as i32
            }
            Err(e) => {
                sess.set_error(e);
                -1
            }
        }
    }));
    res.unwrap_or(-1)
}

/// Clears the target and draft KV caches on `session`.
///
/// # Safety
/// `session` must be null or a valid `KronosMobileSession` pointer.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_reset(session: *mut KronosMobileSession) {
    if session.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Validated non-null exclusive session pointer.
        let sess = unsafe { &mut *session };
        sess.model.reset_cache();
        sess.pending_token = None;
        sess.raw_bytes.clear();
        sess.emitted_bytes = 0;
        sess.clear_error();
    }));
}

unsafe fn write_c_utf8_buf(s: &str, out_buf: *mut c_char, out_cap: usize) {
    if out_buf.is_null() || out_cap == 0 {
        return;
    }
    let clean = s.replace('\0', "");
    let max_copy = clean.len().min(out_cap - 1);
    let mut end = max_copy;
    while end > 0 && !clean.is_char_boundary(end) {
        end -= 1;
    }
    // SAFETY: Caller supplies writable buffer of at least `out_cap` bytes (`end + 1 <= out_cap`).
    unsafe {
        std::ptr::copy_nonoverlapping(clean.as_ptr().cast::<c_char>(), out_buf, end);
        *out_buf.add(end) = 0;
    }
}

/// Ingests raw file or image bytes (`PDF`, `DOCX`, `XLSX`, `ZIP`, `CSV`, `JSON`, code, `PNG`, `JPEG`)
/// into an inline prompt context block and optional `.kronos_media/` workspace asset.
/// Returns `0` on success, `-1` on error.
///
/// # Safety
/// Pointer arguments must either be valid for their given lengths/capacities or null where optional.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn kronos_ingest_attachment_bytes(
    filename_utf8: *const c_char,
    mime_utf8: *const c_char,
    bytes_ptr: *const u8,
    bytes_len: usize,
    workdir_utf8: *const c_char,
    slot: usize,
    out_badge_utf8: *mut c_char,
    out_badge_cap: usize,
    out_context_utf8: *mut c_char,
    out_context_cap: usize,
    out_saved_rel_utf8: *mut c_char,
    out_saved_rel_cap: usize,
) -> i32 {
    if bytes_ptr.is_null() && bytes_len > 0 {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let name = if filename_utf8.is_null() {
            "attachment"
        } else {
            // SAFETY: Validated non-null NUL-terminated string pointer.
            unsafe { CStr::from_ptr(filename_utf8) }
                .to_str()
                .unwrap_or("attachment")
        };
        let mime = if mime_utf8.is_null() {
            ""
        } else {
            // SAFETY: Validated non-null NUL-terminated string pointer.
            unsafe { CStr::from_ptr(mime_utf8) }.to_str().unwrap_or("")
        };
        let bytes: &[u8] = if bytes_len == 0 {
            &[]
        } else {
            // SAFETY: Validated non-null pointer with length `bytes_len`.
            unsafe { std::slice::from_raw_parts(bytes_ptr, bytes_len) }
        };
        let workdir_opt = if workdir_utf8.is_null() {
            None
        } else {
            // SAFETY: Validated non-null NUL-terminated string pointer.
            unsafe { CStr::from_ptr(workdir_utf8) }
                .to_str()
                .ok()
                .filter(|s| !s.trim().is_empty())
                .map(Path::new)
        };
        match crate::runtime::attachments::ingest_attachment_bytes(name, mime, bytes, workdir_opt, slot) {
            Ok(ing) => {
                // SAFETY: Caller-provided output buffers and capacities.
                unsafe {
                    write_c_utf8_buf(&ing.badge, out_badge_utf8, out_badge_cap);
                    write_c_utf8_buf(&ing.prompt_context, out_context_utf8, out_context_cap);
                    write_c_utf8_buf(
                        ing.saved_rel_path.as_deref().unwrap_or(""),
                        out_saved_rel_utf8,
                        out_saved_rel_cap,
                    );
                }
                0
            }
            Err(_) => -1,
        }
    }));
    res.unwrap_or(-1)
}

/// Runs a sandboxed multi-step workspace agent turn (`write_file`, `edit_file`, `read_file`,
/// `list_dir`, `search_files`) inside `workdir_utf8` with `Kronos-Jev` verification.
/// Streams live prose and `[tool: ...]` status lines to `on_piece_cb` and writes the final
/// response into `out_final_utf8`. Returns steps executed (`>= 1`), or `-1` on error.
///
/// # Safety
/// `session`, `workdir_utf8`, and `messages_json_utf8` must be valid non-null pointers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn kronos_session_run_agent_turn(
    session: *mut KronosMobileSession,
    workdir_utf8: *const c_char,
    messages_json_utf8: *const c_char,
    max_steps: usize,
    max_tokens: usize,
    temperature: f32,
    on_piece_cb: KronosStreamCallback,
    user_data: *mut c_void,
    out_final_utf8: *mut c_char,
    out_final_cap: usize,
) -> i32 {
    if session.is_null() {
        return -1;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Validated non-null exclusive session pointer.
        let sess = unsafe { &mut *session };
        // SAFETY: Validated non-null C string pointers per FFI contract.
        let workdir_str = match unsafe { cstr_to_str(workdir_utf8) } {
            Ok(v) => v,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        let raw_json = match unsafe { cstr_to_str(messages_json_utf8) } {
            Ok(v) => v,
            Err(e) => {
                sess.set_error(e);
                return -1;
            }
        };
        let workdir = Path::new(workdir_str);
        if let Err(e) = std::fs::create_dir_all(workdir) {
            sess.set_error(format!("failed to create workspace dir: {e}"));
            return -1;
        }
        let items = match Json::parse(raw_json) {
            Ok(Json::Arr(arr)) => arr,
            Ok(_) => {
                sess.set_error("messages_json_utf8 must be a JSON array");
                return -1;
            }
            Err(e) => {
                sess.set_error(format!("invalid messages_json_utf8: {e}"));
                return -1;
            }
        };
        let mut history = Vec::with_capacity(items.len() + 1);
        history.push(Message::new(
            "system",
            &crate::runtime::agent::build_unified_system_prompt(),
        ));
        for item in &items {
            let role = item
                .get("role")
                .and_then(Json::as_str)
                .unwrap_or("")
                .trim();
            if role == "system" || role.is_empty() {
                continue;
            }
            let content = item.get("content").and_then(Json::as_str).unwrap_or("");
            if !content.is_empty() {
                history.push(Message::new(role, content));
            }
        }
        let tools: Vec<String> = crate::runtime::agent::tool_definitions_json()
            .into_iter()
            .filter(|t| !t.contains("\"run_command\"") && !t.contains("\"switch_workspace\""))
            .collect();
        let cfg = crate::runtime::agent::AgentTurnConfig {
            workdir: workdir.to_path_buf(),
            params: GenParams {
                temperature: temperature.max(0.0),
                max_tokens: max_tokens.max(128),
                ..GenParams::default()
            },
            max_steps: max_steps.clamp(1, 8),
            cmd_timeout_secs: 10,
            expect_tool_call: true,
            prompt_tools: Vec::new(),
        };
        let mut steps_done = 1i32;
        let emit = |piece: &str| -> bool {
            let Some(cb) = on_piece_cb else {
                return true;
            };
            let Ok(cs) = CString::new(piece.replace('\0', "")) else {
                return true;
            };
            // SAFETY: `cs` is alive for the duration of the callback.
            unsafe { cb(cs.as_ptr(), cs.as_bytes().len(), user_data) }
        };
        let turn_res =
            crate::runtime::agent::run_agent_turn(&mut sess.model, &mut history, &tools, &cfg, |ev| {
                match ev {
                    crate::runtime::agent::AgentEvent::StepStart { step, .. } => {
                        steps_done = step as i32;
                        true
                    }
                    crate::runtime::agent::AgentEvent::TextDelta(d) => emit(d),
                    crate::runtime::agent::AgentEvent::ToolStart { call, .. } => {
                        emit(&format!("\n[tool {}]\n", call.name))
                    }
                    crate::runtime::agent::AgentEvent::ToolDone { call, outcome, .. } => {
                        let first_line = outcome.output.lines().next().unwrap_or("done");
                        emit(&format!("[{} -> {}]\n", call.name, first_line))
                    }
                    crate::runtime::agent::AgentEvent::Done { steps, .. } => {
                        steps_done = steps as i32;
                        true
                    }
                    _ => true,
                }
            });
        match turn_res {
            Ok(final_txt) => {
                // SAFETY: Caller-provided output buffer and capacity.
                unsafe {
                    write_c_utf8_buf(&final_txt, out_final_utf8, out_final_cap);
                }
                sess.clear_error();
                steps_done.max(1)
            }
            Err(e) => {
                sess.set_error(e);
                -1
            }
        }
    }));
    res.unwrap_or(-1)
}

/// Returns resident model weight + active KV-cache bytes for `session`.
///
/// # Safety
/// `session` must be null or a valid `KronosMobileSession` pointer.
#[no_mangle]
pub unsafe extern "C" fn kronos_session_resident_bytes(
    session: *const KronosMobileSession,
) -> usize {
    if session.is_null() {
        return 0;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Shared read-only reference to live session.
        let sess = unsafe { &*session };
        sess.model
            .resident_weight_bytes()
            .saturating_add(sess.model.kv_cache_bytes())
    }));
    res.unwrap_or(0)
}

/// Returns live Mach `task_info` physical resident set size (RSS) of the host process in bytes.
#[no_mangle]
pub extern "C" fn kronos_session_process_rss_bytes() -> usize {
    current_resident_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    unsafe extern "C" fn collect_stream_cb(
        ptr: *const c_char,
        len: usize,
        user_data: *mut c_void,
    ) -> bool {
        if ptr.is_null() || user_data.is_null() {
            return false;
        }
        // SAFETY: `ptr` points to `len` valid UTF-8 bytes and `user_data` points to `String`.
        unsafe {
            let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len);
            let out = &mut *user_data.cast::<String>();
            out.push_str(&String::from_utf8_lossy(bytes));
        }
        true
    }

    #[test]
    fn test_mobile_ffi_lifecycle_jev_prefill_step_and_budget() {
        // Null-pointer safety invariants.
        // SAFETY: Null pointers are explicitly supported by the FFI contract.
        unsafe {
            assert!(kronos_session_create(std::ptr::null(), c"nf4g128".as_ptr(), 256, 2, 0).is_null());
            assert_eq!(kronos_session_resident_bytes(std::ptr::null()), 0);
            kronos_session_reset(std::ptr::null_mut());
            kronos_session_destroy(std::ptr::null_mut());
        }
        assert!(kronos_session_process_rss_bytes() > 0);

        let model_dir = Path::new("models/Qwen2.5-Coder-0.5B-Instruct");
        if !model_dir.join("kronos-nf4g128.k2tb").exists() {
            return;
        }
        let dir_c = CString::new(model_dir.to_str().unwrap()).unwrap();
        let mode_c = CString::new("nf4g128").unwrap();

        // Enforce 4900 MiB iOS `Mobile5G` memory budget via C FFI.
        // SAFETY: Valid CStrings and exclusive session ownership throughout test.
        unsafe {
            let sess = kronos_session_create(dir_c.as_ptr(), mode_c.as_ptr(), 2048, 2, 4900);
            assert!(!sess.is_null(), "kronos_session_create returned null");
            assert_eq!(
                MemoryBudgetProfile::active().tier,
                crate::model::memory_budget::MemoryTier::Mobile5G
            );

            // Configure Hadamard Q4 sliding-window KV (`128` window + `4` sinks).
            assert_eq!(kronos_session_configure_kv(sess, 4, 128, 4), 0);

            let res_bytes = kronos_session_resident_bytes(sess);
            assert!(
                res_bytes > 100 * 1024 * 1024 && res_bytes < 512 * 1024 * 1024,
                "unexpected 0.5B resident bytes: {res_bytes}"
            );

            // Single-forward-pass `Kronos-Jev` decision (`0` decode tokens).
            let mut probs = [0.0f32; 3];
            let mut entropy = -1.0f32;
            let mut confidence = -1.0f32;
            let selected = kronos_session_jev_decide(
                sess,
                c"Classify whether the user wants code editing (Agent), macOS app launch (Computer), or plain Q&A (Chat).".as_ptr(),
                c"Edit src/main.rs to fix the compiler error and run tests.".as_ptr(),
                c"[\"Chat\",\"Computer\",\"Agent\"]".as_ptr(),
                1.0,
                probs.as_mut_ptr(),
                probs.len(),
                &mut entropy,
                &mut confidence,
            );
            assert!(
                (0..3).contains(&selected),
                "jev_decide failed: {}",
                CStr::from_ptr(kronos_session_last_error(sess)).to_string_lossy()
            );
            let prob_sum: f32 = probs.iter().sum();
            assert!((prob_sum - 1.0).abs() < 1e-4, "unnormalized probs: {probs:?}");
            assert!(entropy >= 0.0 && (0.0..=1.0).contains(&confidence));

            // Stateful `prefill` + `step` token-by-token loop.
            let first_tok = kronos_session_prefill(
                sess,
                c"Reply concisely in one short sentence.".as_ptr(),
                c"What is 2 + 2?".as_ptr(),
            );
            assert!(first_tok >= 0, "prefill failed");
            let mut stepped_text = String::new();
            let mut steps = 0usize;
            let mut piece_buf = [0 as c_char; 128];
            while steps < 16 {
                let mut tok_id = -1i32;
                let rc = kronos_session_step(
                    sess,
                    piece_buf.as_mut_ptr(),
                    piece_buf.len(),
                    &mut tok_id,
                );
                assert!(rc >= 0, "step error");
                if rc == 0 {
                    break;
                }
                assert!(tok_id >= 0);
                stepped_text.push_str(&CStr::from_ptr(piece_buf.as_ptr()).to_string_lossy());
                steps += 1;
            }
            assert!(steps > 0 && !stepped_text.trim().is_empty());

            // Streaming callback generation API.
            let mut cb_text = String::new();
            let n_gen = kronos_session_generate(
                sess,
                c"Reply with only the number.".as_ptr(),
                c"What is 3 * 3?".as_ptr(),
                12,
                0.0,
                Some(collect_stream_cb),
                (&mut cb_text as *mut String).cast::<c_void>(),
            );
            assert!(n_gen > 0 && cb_text.contains('9'), "unexpected reply: {cb_text:?}");

            // File/document byte ingestion FFI check.
            let sample_csv = b"model,bits,tok_s\nkronos,4.16,119.2\n";
            let mut badge_buf = [0 as c_char; 256];
            let mut ctx_buf = [0 as c_char; 2048];
            let mut rel_buf = [0 as c_char; 256];
            let ing_rc = kronos_ingest_attachment_bytes(
                c"bench.csv".as_ptr(),
                c"text/csv".as_ptr(),
                sample_csv.as_ptr(),
                sample_csv.len(),
                std::ptr::null(),
                1,
                badge_buf.as_mut_ptr(),
                badge_buf.len(),
                ctx_buf.as_mut_ptr(),
                ctx_buf.len(),
                rel_buf.as_mut_ptr(),
                rel_buf.len(),
            );
            assert_eq!(ing_rc, 0);
            let ing_ctx = CStr::from_ptr(ctx_buf.as_ptr()).to_string_lossy();
            assert!(ing_ctx.contains("kronos") && ing_ctx.contains("119.2"));

            let t1_json = CString::new(
                r#"[{"role":"system","content":"Reply with only the number."},{"role":"user","content":"What is 3 * 3?"}]"#,
            )
            .unwrap();
            let mut t1_cpu = String::new();
            let mut pf_cpu = 0.0f64;
            let mut dc_cpu = 0.0f64;
            let mut pft_cpu = 0usize;
            kronos_session_reset(sess);
            let n1_cpu = kronos_session_generate_chat_json(
                sess,
                t1_json.as_ptr(),
                16,
                0.0,
                Some(collect_stream_cb),
                (&mut t1_cpu as *mut String).cast::<c_void>(),
                &mut pf_cpu,
                &mut dc_cpu,
                &mut pft_cpu,
            );
            assert!(n1_cpu > 0 && t1_cpu.contains('9'), "cpu turn1: {t1_cpu:?}");

            // Apple Metal GPU backend: verify `jev_decide` (cold + hot KV prefix) and
            // multi-turn `generate_chat_json` deterministic prefill + KV reuse (`dim=896`).
            #[cfg(all(any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
            if kronos_session_enable_metal(sess) == 0 {
                kronos_session_reset(sess);
                let mut m_probs = [0.0f32; 3];
                let mut m_ent = 0.0f32;
                let mut m_conf = 0.0f32;
                let j_sys = c"Classify whether the user wants code editing (Agent), device control (Computer), or Q&A (Chat).";
                let m_sel_cold = kronos_session_jev_decide(
                    sess,
                    j_sys.as_ptr(),
                    c"Edit src/main.rs to fix the compiler error.".as_ptr(),
                    c"[\"Chat\",\"Computer\",\"Agent\"]".as_ptr(),
                    1.0,
                    m_probs.as_mut_ptr(),
                    3,
                    &mut m_ent,
                    &mut m_conf,
                );
                assert!(
                    (0..3).contains(&m_sel_cold),
                    "metal jev_decide cold failed: {}",
                    CStr::from_ptr(kronos_session_last_error(sess)).to_string_lossy()
                );
                let m_sel_hot = kronos_session_jev_decide(
                    sess,
                    j_sys.as_ptr(),
                    c"Explain Walsh-Hadamard rotation in 4-bit quantization.".as_ptr(),
                    c"[\"Chat\",\"Computer\",\"Agent\"]".as_ptr(),
                    1.0,
                    m_probs.as_mut_ptr(),
                    3,
                    &mut m_ent,
                    &mut m_conf,
                );
                assert!((0..3).contains(&m_sel_hot), "metal jev_decide hot failed");

                let mut t1_out_a = String::new();
                let mut pf1 = 0.0f64;
                let mut dc1 = 0.0f64;
                let mut pft1 = 0usize;
                kronos_session_reset(sess);
                let n1_a = kronos_session_generate_chat_json(
                    sess,
                    t1_json.as_ptr(),
                    16,
                    0.0,
                    Some(collect_stream_cb),
                    (&mut t1_out_a as *mut String).cast::<c_void>(),
                    &mut pf1,
                    &mut dc1,
                    &mut pft1,
                );
                assert!(n1_a > 0 && t1_out_a.contains('9'), "metal turn1 a: {t1_out_a:?}");

                // Repeat after reset to assert deterministic `rmsnorm_fwht64` prefill (no WAR race).
                let mut t1_out_b = String::new();
                kronos_session_reset(sess);
                let n1_b = kronos_session_generate_chat_json(
                    sess,
                    t1_json.as_ptr(),
                    16,
                    0.0,
                    Some(collect_stream_cb),
                    (&mut t1_out_b as *mut String).cast::<c_void>(),
                    &mut pf1,
                    &mut dc1,
                    &mut pft1,
                );
                assert_eq!(n1_a, n1_b);
                assert_eq!(t1_out_a, t1_out_b, "non-deterministic Metal prefill");

                // Turn 2 reuses the Turn 1 KV prefix (`pft2 < pft1`).
                let t2_json = CString::new(format!(
                    r#"[{{"role":"system","content":"Reply with only the number."}},{{"role":"user","content":"What is 3 * 3?"}},{{"role":"assistant","content":{}}},{{"role":"user","content":"What is 2 + 2?"}}]"#,
                    crate::util::json::quote(t1_out_b.trim())
                ))
                .unwrap();
                let mut t2_out = String::new();
                let mut pf2 = 0.0f64;
                let mut dc2 = 0.0f64;
                let mut pft2 = 0usize;
                let n2 = kronos_session_generate_chat_json(
                    sess,
                    t2_json.as_ptr(),
                    16,
                    0.0,
                    Some(collect_stream_cb),
                    (&mut t2_out as *mut String).cast::<c_void>(),
                    &mut pf2,
                    &mut dc2,
                    &mut pft2,
                );
                assert!(n2 > 0 && t2_out.contains('4'), "metal turn2: {t2_out:?}");
                assert!(
                    pft2 > 0 && pft2 < pft1,
                    "expected Turn 2 KV prefix reuse: pft1={pft1}, pft2={pft2}"
                );

                // Verify 0.5B mobile creator profile grounding (he/him, Software Engineer > 10 years, never musician).
                assert_eq!(kronos_session_configure_kv(sess, 8, 0, 0), 0);
                kronos_session_reset(sess);
                let sys = crate::runtime::chat::DEFAULT_SYSTEM_QWEN;
                let facts = crate::runtime::chat::CREATOR_PUBLIC_PROFILE;
                let q1 = format!("{facts}\n\nUser question: Who is Melody?");
                let q1_json = CString::new(format!(
                    r#"[{{"role":"system","content":{}}},{{"role":"user","content":{}}}]"#,
                    crate::util::json::quote(sys),
                    crate::util::json::quote(&q1)
                ))
                .unwrap();
                let mut prof_out = String::new();
                let np1 = kronos_session_generate_chat_json(
                    sess,
                    q1_json.as_ptr(),
                    140,
                    0.0,
                    Some(collect_stream_cb),
                    (&mut prof_out as *mut String).cast::<c_void>(),
                    &mut pf1,
                    &mut dc1,
                    &mut pft1,
                );
                let prof_low = prof_out.to_ascii_lowercase();
                assert!(
                    np1 > 0 && prof_low.contains("software engineer") && !prof_low.contains("musician") && !prof_low.contains(" she ") && !prof_low.contains(" her "),
                    "unexpected 0.5B creator profile answer: {prof_out:?}"
                );

                let q2 = format!("{facts}\n\nUser question: What's his email and GitHub?");
                let q2_json = CString::new(format!(
                    r#"[{{"role":"system","content":{}}},{{"role":"user","content":{}}},{{"role":"assistant","content":{}}},{{"role":"user","content":{}}}]"#,
                    crate::util::json::quote(sys),
                    crate::util::json::quote(&q1),
                    crate::util::json::quote(prof_out.trim()),
                    crate::util::json::quote(&q2)
                ))
                .unwrap();
                let mut contact_out = String::new();
                let np2 = kronos_session_generate_chat_json(
                    sess,
                    q2_json.as_ptr(),
                    120,
                    0.0,
                    Some(collect_stream_cb),
                    (&mut contact_out as *mut String).cast::<c_void>(),
                    &mut pf2,
                    &mut dc2,
                    &mut pft2,
                );
                let contact_low = contact_out.to_ascii_lowercase();
                assert!(
                    np2 > 0
                        && contact_low.contains("melodyleonard7@gmail.com")
                        && contact_low.contains("github.com/melodyleonard"),
                    "unexpected 0.5B contact answer: {contact_out:?}"
                );

                assert_eq!(kronos_session_disable_metal(sess), 0);
            }

            kronos_session_reset(sess);
            kronos_session_destroy(sess);
        }
        MemoryBudgetProfile::set_global_budget_override_bytes(0);
    }
}
