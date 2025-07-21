from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class AnthropicMessage:
    role: Literal["user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]


class Usage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def __getitem__(self, key: str):
        if key == "input_tokens":
            return self.input_tokens
        elif key == "output_tokens":
            return self.output_tokens
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return (
            f"Usage(input_tokens={self.input_tokens}, "
            f"output_tokens={self.output_tokens})"
        )


class ContentBlock:
    def __init__(self, type: str, text: str = ""):
        self.type = type
        self.text = text

    def __getitem__(self, key: str):
        if key == "type":
            return self.type
        elif key == "text":
            return self.text
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return f"ContentBlock(type='{self.type}', text='{self.text}')"


class Message:
    def __init__(
        self,
        id: str,
        content: List["ContentBlockType"],
        model: str,
        role: str = "assistant",
        stop_reason: Optional[str] = None,
        stop_sequence: Optional[str] = None,
        type: str = "message",
        usage: Optional[Usage] = None,
    ):
        self.id = id
        self.type = type
        self.role = role
        self.content = content
        self.model = model
        self.stop_reason = stop_reason
        self.stop_sequence = stop_sequence
        self.usage = usage

    def __getitem__(self, key: str):
        if key == "id":
            return self.id
        elif key == "type":
            return self.type
        elif key == "role":
            return self.role
        elif key == "content":
            return self.content
        elif key == "model":
            return self.model
        elif key == "stop_reason":
            return self.stop_reason
        elif key == "stop_sequence":
            return self.stop_sequence
        elif key == "usage":
            return self.usage
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return (
            f"Message(id='{self.id}', content={self.content}, "
            f"model='{self.model}', stop_reason='{self.stop_reason}')"
        )


@dataclass
class AnthropicUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class AnthropicResponse:
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    content: List[Dict[str, Any]]
    model: str
    stop_reason: Optional[str]
    stop_sequence: Optional[str]
    usage: AnthropicUsage


@dataclass
class OpenAIMessage:
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class OpenAIChoice:
    index: int
    message: OpenAIMessage
    logprobs: Optional[Any]
    finish_reason: Optional[str]


@dataclass
class OpenAIUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class OpenAIResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage
    system_fingerprint: Optional[str] = None


AnthropicCreateParams = Dict[str, Any]
OpenAICreateParams = Dict[str, Any]


# Tool-related types
@dataclass
class ToolUse:
    id: str
    name: str
    input: Dict[str, Any]
    type: str = "tool_use"

    def __getitem__(self, key: str):
        if key == "id":
            return self.id
        elif key == "name":
            return self.name
        elif key == "input":
            return self.input
        elif key == "type":
            return self.type
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return f"ToolUse(id='{self.id}', name='{self.name}', input={self.input})"


@dataclass
class ToolResult:
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]]
    is_error: bool = False
    type: str = "tool_result"

    def __getitem__(self, key: str):
        if key == "tool_use_id":
            return self.tool_use_id
        elif key == "content":
            return self.content
        elif key == "is_error":
            return self.is_error
        elif key == "type":
            return self.type
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return (
            f"ToolResult(tool_use_id='{self.tool_use_id}', "
            f"content={self.content}, is_error={self.is_error})"
        )


# Union type for content blocks (after ToolUse is defined)
ContentBlockType = Union[ContentBlock, ToolUse]


@dataclass
class AnthropicTool:
    name: str
    description: str
    input_schema: Dict[str, Any]


# Streaming types
class StreamingContentBlock:
    def __init__(self, type: str, text: str = ""):
        self.type = type
        self.text = text

    def __getitem__(self, key: str):
        if key == "type":
            return self.type
        elif key == "text":
            return self.text
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return f"StreamingContentBlock(type='{self.type}', text='{self.text}')"


class StreamingMessage:
    def __init__(
        self,
        id: str = "",
        type: str = "message",
        role: str = "assistant",
        content: Optional[List[Union[ContentBlock, ToolUse]]] = None,
        model: str = "",
        stop_reason: Optional[str] = None,
        stop_sequence: Optional[str] = None,
        usage: Optional[Usage] = None,
    ):
        self.id = id
        self.type = type
        self.role = role
        self.content = content or []
        self.model = model
        self.stop_reason = stop_reason
        self.stop_sequence = stop_sequence
        self.usage = usage

    def __getitem__(self, key: str):
        if key == "id":
            return self.id
        elif key == "type":
            return self.type
        elif key == "role":
            return self.role
        elif key == "content":
            return self.content
        elif key == "model":
            return self.model
        elif key == "stop_reason":
            return self.stop_reason
        elif key == "stop_sequence":
            return self.stop_sequence
        elif key == "usage":
            return self.usage
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return (
            f"StreamingMessage(id='{self.id}', content={self.content}, "
            f"model='{self.model}', stop_reason='{self.stop_reason}')"
        )


class MessageDelta:
    def __init__(
        self,
        type: str = "message_delta",
        delta: Optional[Dict[str, Any]] = None,
        usage: Optional[Usage] = None,
    ):
        self.type = type
        self.delta = delta or {}
        self.usage = usage

    def __getitem__(self, key: str):
        if key == "type":
            return self.type
        elif key == "delta":
            return self.delta
        elif key == "usage":
            return self.usage
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return (
            f"MessageDelta(type='{self.type}', delta={self.delta}, usage={self.usage})"
        )


class ContentBlockDelta:
    def __init__(
        self,
        type: str = "content_block_delta",
        index: int = 0,
        delta: Optional[Dict[str, Any]] = None,
    ):
        self.type = type
        self.index = index
        self.delta = delta or {}

    def __getitem__(self, key: str):
        if key == "type":
            return self.type
        elif key == "index":
            return self.index
        elif key == "delta":
            return self.delta
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return (
            f"ContentBlockDelta(type='{self.type}', index={self.index}, "
            f"delta={self.delta})"
        )


class ContentBlockStart:
    def __init__(
        self,
        type: str = "content_block_start",
        index: int = 0,
        content_block: Optional[Union[ContentBlock, ToolUse]] = None,
    ):
        self.type = type
        self.index = index
        self.content_block = content_block

    def __getitem__(self, key: str):
        if key == "type":
            return self.type
        elif key == "index":
            return self.index
        elif key == "content_block":
            return self.content_block
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return (
            f"ContentBlockStart(type='{self.type}', index={self.index}, "
            f"content_block={self.content_block})"
        )


class ContentBlockStop:
    def __init__(
        self,
        type: str = "content_block_stop",
        index: int = 0,
    ):
        self.type = type
        self.index = index

    def __getitem__(self, key: str):
        if key == "type":
            return self.type
        elif key == "index":
            return self.index
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return f"ContentBlockStop(type='{self.type}', index={self.index})"


class MessageStart:
    def __init__(
        self,
        type: str = "message_start",
        message: Optional[StreamingMessage] = None,
    ):
        self.type = type
        self.message = message or StreamingMessage()

    def __getitem__(self, key: str):
        if key == "type":
            return self.type
        elif key == "message":
            return self.message
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return f"MessageStart(type='{self.type}', message={self.message})"


class MessageStop:
    def __init__(self, type: str = "message_stop"):
        self.type = type

    def __getitem__(self, key: str):
        if key == "type":
            return self.type
        else:
            raise KeyError(f"'{key}' is not a valid key")

    def __repr__(self):
        return f"MessageStop(type='{self.type}')"


# Type aliases for streaming events
StreamingEvent = Union[
    MessageStart,
    ContentBlockStart,
    ContentBlockDelta,
    ContentBlockStop,
    MessageDelta,
    MessageStop,
]


StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
OpenAIFinishReason = Literal[
    "stop", "length", "function_call", "tool_calls", "content_filter"
]


ANTHROPIC_TO_OPENAI_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


OPENAI_TO_ANTHROPIC_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "function_call": "tool_use",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}
