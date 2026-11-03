//! JNI entrypoints for `com.kronos.mobile.KronosEngine`, over the C API in `mobile_ffi`.
//!
//! Strings cross as UTF-8 `byte[]` because JNI's modified UTF-8 cannot carry emoji.

use super::mobile_ffi::*;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Mutex;

type Env = *mut *const *const c_void;
type Obj = *mut c_void;

static CREATE_ERROR: Mutex<String> = Mutex::new(String::new());

// Slots in the JNINativeInterface table.
const EXCEPTION_CLEAR: usize = 17;
const DELETE_LOCAL_REF: usize = 23;
const GET_OBJECT_CLASS: usize = 31;
const GET_METHOD_ID: usize = 33;
const CALL_BOOLEAN_METHOD_A: usize = 39;
const GET_ARRAY_LENGTH: usize = 171;
const NEW_BYTE_ARRAY: usize = 176;
const GET_BYTE_ARRAY_REGION: usize = 200;
const SET_BYTE_ARRAY_REGION: usize = 208;
const SET_DOUBLE_ARRAY_REGION: usize = 214;
const EXCEPTION_CHECK: usize = 228;

/// # Safety
/// `env` must be the JNIEnv of the current thread and `F` the slot's real signature.
unsafe fn slot<F: Copy>(env: Env, index: usize) -> F {
    // SAFETY: Slots are fixed by the JNI spec and every function pointer is the same size.
    unsafe { std::mem::transmute_copy(&*(*env).add(index)) }
}

unsafe fn cstring(env: Env, bytes: Obj) -> CString {
    // SAFETY: `bytes` is a live byte[] and the buffer holds exactly its length.
    unsafe {
        let len: extern "C" fn(Env, Obj) -> i32 = slot(env, GET_ARRAY_LENGTH);
        let get: extern "C" fn(Env, Obj, i32, i32, *mut u8) = slot(env, GET_BYTE_ARRAY_REGION);
        let n = len(env, bytes);
        let mut buf = vec![0u8; n as usize];
        get(env, bytes, 0, n, buf.as_mut_ptr());
        buf.retain(|b| *b != 0);
        CString::new(buf).unwrap_or_default()
    }
}

unsafe fn byte_array(env: Env, s: &[u8]) -> Obj {
    // SAFETY: The new array is exactly `s.len()` long.
    unsafe {
        let new: extern "C" fn(Env, i32) -> Obj = slot(env, NEW_BYTE_ARRAY);
        let set: extern "C" fn(Env, Obj, i32, i32, *const u8) = slot(env, SET_BYTE_ARRAY_REGION);
        let arr = new(env, s.len() as i32);
        if !arr.is_null() {
            set(env, arr, 0, s.len() as i32, s.as_ptr());
        }
        arr
    }
}

struct Stream {
    env: Env,
    target: Obj,
    method: *mut c_void,
}

unsafe extern "C" fn on_piece(ptr: *const c_char, len: usize, user: *mut c_void) -> bool {
    // SAFETY: `user` is the `Stream` on `nativeGenerate`'s stack, on the same thread.
    unsafe {
        let st = &*(user as *const Stream);
        let piece = byte_array(st.env, std::slice::from_raw_parts(ptr as *const u8, len));
        let call: extern "C" fn(Env, Obj, *mut c_void, *const u64) -> u8 = slot(st.env, CALL_BOOLEAN_METHOD_A);
        let arg = piece as u64;
        let keep = call(st.env, st.target, st.method, &arg);
        let del: extern "C" fn(Env, Obj) = slot(st.env, DELETE_LOCAL_REF);
        del(st.env, piece);
        let pending: extern "C" fn(Env) -> u8 = slot(st.env, EXCEPTION_CHECK);
        if pending(st.env) != 0 {
            let clear: extern "C" fn(Env) = slot(st.env, EXCEPTION_CLEAR);
            clear(st.env);
            return false;
        }
        keep != 0
    }
}

/// # Safety
/// Called by the JVM with live `byte[]` arguments.
#[no_mangle]
pub unsafe extern "C" fn Java_com_kronos_mobile_KronosEngine_nativeCreate(
    env: Env, _class: Obj, dir: Obj, mode: Obj, ctx: i32, threads: i32, budget_mb: i32,
) -> i64 {
    // SAFETY: Arguments come from the JVM per this function's contract.
    unsafe {
        let (dir, mode) = (cstring(env, dir), cstring(env, mode));
        match create_session(dir.as_ptr(), mode.as_ptr(), ctx as usize, threads as usize, budget_mb as usize) {
            Ok(s) => Box::into_raw(Box::new(s)) as i64,
            Err(e) => {
                *CREATE_ERROR.lock().unwrap_or_else(|p| p.into_inner()) = e;
                0
            }
        }
    }
}

/// # Safety
/// Called by the JVM.
#[no_mangle]
pub unsafe extern "C" fn Java_com_kronos_mobile_KronosEngine_nativeCreateError(env: Env, _class: Obj) -> Obj {
    let e = CREATE_ERROR.lock().unwrap_or_else(|p| p.into_inner()).clone();
    // SAFETY: `env` is the current thread's JNIEnv.
    unsafe { byte_array(env, e.as_bytes()) }
}

/// # Safety
/// `handle` must come from `nativeCreate` and not be destroyed.
#[no_mangle]
pub unsafe extern "C" fn Java_com_kronos_mobile_KronosEngine_nativeConfigureKv(
    _env: Env, _class: Obj, handle: i64, bits: i32, window: i32, sink: i32,
) -> i32 {
    // SAFETY: Handle validity per this function's contract.
    unsafe { kronos_session_configure_kv(handle as *mut _, bits as u8, window as usize, sink as usize) }
}

/// Streams through `callback.onPiece(byte[]): Boolean`. `metrics` receives prefill ms, decode ms
/// and prefilled tokens. Returns generated tokens, or -1.
///
/// # Safety
/// `handle` must be live; the other arguments come from the JVM.
#[no_mangle]
pub unsafe extern "C" fn Java_com_kronos_mobile_KronosEngine_nativeGenerate(
    env: Env, _class: Obj, handle: i64, messages_json: Obj, max_tokens: i32, temperature: f32, callback: Obj, metrics: Obj,
) -> i32 {
    // SAFETY: Handle validity per this function's contract; JNI calls run on this thread.
    unsafe {
        let json = cstring(env, messages_json);
        let class_of: extern "C" fn(Env, Obj) -> Obj = slot(env, GET_OBJECT_CLASS);
        let method_id: extern "C" fn(Env, Obj, *const c_char, *const c_char) -> *mut c_void = slot(env, GET_METHOD_ID);
        let class = class_of(env, callback);
        let method = method_id(env, class, c"onPiece".as_ptr(), c"([B)Z".as_ptr());
        if method.is_null() {
            return -1;
        }
        let stream = Stream { env, target: callback, method };
        let (mut prefill, mut decode, mut tokens) = (0f64, 0f64, 0usize);
        let n = kronos_session_generate_chat_json(
            handle as *mut _,
            json.as_ptr(),
            max_tokens as usize,
            temperature,
            Some(on_piece),
            &stream as *const Stream as *mut c_void,
            &mut prefill,
            &mut decode,
            &mut tokens,
        );
        let set: extern "C" fn(Env, Obj, i32, i32, *const f64) = slot(env, SET_DOUBLE_ARRAY_REGION);
        let out = [prefill, decode, tokens as f64];
        set(env, metrics, 0, 3, out.as_ptr());
        n
    }
}

/// # Safety
/// `handle` must be live.
#[no_mangle]
pub unsafe extern "C" fn Java_com_kronos_mobile_KronosEngine_nativeLastError(env: Env, _class: Obj, handle: i64) -> Obj {
    // SAFETY: Handle validity per this function's contract; the error string outlives this call.
    unsafe {
        let p = kronos_session_last_error(handle as *const _);
        let bytes = if p.is_null() { &[][..] } else { CStr::from_ptr(p).to_bytes() };
        byte_array(env, bytes)
    }
}

/// # Safety
/// `handle` must be live.
#[no_mangle]
pub unsafe extern "C" fn Java_com_kronos_mobile_KronosEngine_nativeReset(_env: Env, _class: Obj, handle: i64) {
    // SAFETY: Handle validity per this function's contract.
    unsafe { kronos_session_reset(handle as *mut _) }
}

/// # Safety
/// `handle` must be live and is invalid afterwards.
#[no_mangle]
pub unsafe extern "C" fn Java_com_kronos_mobile_KronosEngine_nativeDestroy(_env: Env, _class: Obj, handle: i64) {
    // SAFETY: Handle validity per this function's contract.
    unsafe { kronos_session_destroy(handle as *mut _) }
}

/// Peak resident bytes of this process.
#[no_mangle]
pub extern "C" fn Java_com_kronos_mobile_KronosEngine_nativePeakRss(_env: Env, _class: Obj) -> i64 {
    kronos_session_process_rss_bytes() as i64
}
