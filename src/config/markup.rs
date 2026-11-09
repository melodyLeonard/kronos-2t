//! Chat template and tool-call markup tokens.

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const END_OF_TEXT: &str = "<|endoftext|>";

// HunYuan (Hy-MT2) template: `<bos>[system<sys_end>]<user>...<assistant>...<eos>`.
pub const HY_BOS: &str = "<｜hy_begin▁of▁sentence｜>";
pub const HY_SYSTEM_END: &str = "<｜hy_place▁holder▁no▁3｜>";
pub const HY_USER: &str = "<｜hy_User｜>";
pub const HY_ASSISTANT: &str = "<｜hy_Assistant｜>";
pub const HY_EOS: &str = "<｜hy_place▁holder▁no▁2｜>";

pub const TOOL_CALL_OPEN: &str = "<tool_call>";
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";
pub const TOOLS_OPEN: &str = "<tools>";
pub const TOOLS_CLOSE: &str = "</tools>";
pub const TOOL_RESPONSE_OPEN: &str = "<tool_response>";
pub const TOOL_RESPONSE_CLOSE: &str = "</tool_response>";
pub const XML_TOOL_OPEN: &str = "<tool ";

pub const CODE_FENCE: &str = "```";
pub const JSON_FENCE: &str = "```json";
