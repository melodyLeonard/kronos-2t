//! HTTP server defaults.

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 18080;
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:18080";

pub const MAX_BODY_BYTES: usize = 32 << 20;
pub const MAX_HEADER_BYTES: usize = 64 << 10;
pub const MAX_REVERT_SNAPSHOT_BYTES: usize = 128 << 10;
