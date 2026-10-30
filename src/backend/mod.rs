//! Hardware backends: Metal GPU and Core ML.

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod metal_gpu;
#[cfg(all(feature = "ane", target_os = "macos"))]
pub mod coreml;
