//! C ABI for mobile hosts.

pub mod mobile_ffi;

#[cfg(target_os = "android")]
pub mod android_jni;
