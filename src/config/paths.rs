//! On-disk state locations under the user home directory.

use std::path::{Path, PathBuf};

pub const STATE_DIR: &str = ".kronos";
pub const UI_STATE_DIR: &str = "ui_state";
pub const SESSIONS_DIR: &str = "sessions";
pub const DEVICE_CONTEXT_DIR: &str = "device_context";
pub const BIN_DIR: &str = "bin";
pub const VOICE_SOCKET_FILE: &str = "voice.sock";

pub const CONVERSATION_MEMORY_FILE: &str = "conversation_memory.jsonl";
pub const LEARNED_MEMORY_FILE: &str = "learned_user_memory.json";
pub const DISTILLED_VAULT_FILE: &str = "distilled_14b_vault.json";

pub fn state_root(home: &Path) -> PathBuf {
    home.join(STATE_DIR)
}

pub fn ui_state_dir(home: &Path) -> PathBuf {
    state_root(home).join(UI_STATE_DIR)
}

pub fn sessions_dir(home: &Path) -> PathBuf {
    ui_state_dir(home).join(SESSIONS_DIR)
}

pub fn device_context_dir(home: &Path) -> PathBuf {
    state_root(home).join(DEVICE_CONTEXT_DIR)
}

pub fn bin_dir(home: &Path) -> PathBuf {
    state_root(home).join(BIN_DIR)
}
