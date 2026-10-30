//! Generation, agent, and context limits.

use super::units::KIB;

/// Minimum new prompt tokens before Neural Engine prefill beats extending the CPU cache.
pub const ANE_MIN_NEW_TOKENS: usize = 32;

/// Maximum draft tokens verified in one target batch (`1 + 7 = 8` tokens).
pub const SPEC_DRAFT_K_GREEDY: usize = 7;
pub const SPEC_DRAFT_K_SAMPLE: usize = 7;
pub const SPEC_DRAFT_K_INIT: usize = 4;

pub const JEV_CACHE_SLOTS: usize = 2;
pub const JEV_PREFIX_CAP: usize = 80;

pub const MAX_READ_BYTES: usize = 24 * KIB;
pub const MAX_CMD_OUTPUT_BYTES: usize = 8 * KIB;
pub const MAX_SEARCH_MATCHES: usize = 60;
pub const MAX_CONSECUTIVE_NO_TOOL_STEPS: usize = 5;

pub const MAX_LEARNED_FACTS: usize = 256;
