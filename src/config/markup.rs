//! Chat template and tool-call markup tokens.

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const END_OF_TEXT: &str = "<|endoftext|>";

pub const TOOL_CALL_OPEN: &str = "<tool_call>";
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";
pub const TOOLS_OPEN: &str = "<tools>";
pub const TOOLS_CLOSE: &str = "</tools>";
pub const TOOL_RESPONSE_OPEN: &str = "<tool_response>";
pub const TOOL_RESPONSE_CLOSE: &str = "</tool_response>";
pub const XML_TOOL_OPEN: &str = "<tool ";

pub const CODE_FENCE: &str = "```";
pub const JSON_FENCE: &str = "```json";
