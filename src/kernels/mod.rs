//! Compute kernels: LUT GEMV, SIMD planes, Hadamard, routing, and threading.

pub mod accel;
pub mod hadamard;
pub mod lut_gemv;
pub mod simd_lut;
pub mod synthesis;
pub mod router;
pub mod ssd;
pub mod thread_pool;
