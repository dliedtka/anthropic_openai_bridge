"""
Tests for types module classes and their methods.
"""

import pytest

from anthropic_openai_bridge.types import (
    ContentBlock,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    Message,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamingMessage,
    ToolResult,
    ToolUse,
    Usage,
)


class TestUsage:
    """Test Usage class."""

    def test_usage_init(self):
        """Test Usage initialization."""
        usage = Usage(input_tokens=10, output_tokens=25)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 25

    def test_usage_dict_access(self):
        """Test Usage dictionary-like access."""
        usage = Usage(input_tokens=15, output_tokens=30)
        assert usage["input_tokens"] == 15
        assert usage["output_tokens"] == 30

    def test_usage_invalid_key(self):
        """Test Usage with invalid key."""
        usage = Usage(input_tokens=10, output_tokens=25)
        with pytest.raises(KeyError, match="'invalid_key' is not a valid key"):
            _ = usage["invalid_key"]

    def test_usage_repr(self):
        """Test Usage string representation."""
        usage = Usage(input_tokens=10, output_tokens=25)
        repr_str = repr(usage)
        assert "Usage(input_tokens=10, output_tokens=25)" in repr_str


class TestContentBlock:
    """Test ContentBlock class."""

    def test_content_block_init(self):
        """Test ContentBlock initialization."""
        block = ContentBlock(type="text", text="Hello world")
        assert block.type == "text"
        assert block.text == "Hello world"

    def test_content_block_default_text(self):
        """Test ContentBlock with default text."""
        block = ContentBlock(type="text")
        assert block.type == "text"
        assert block.text == ""

    def test_content_block_dict_access(self):
        """Test ContentBlock dictionary-like access."""
        block = ContentBlock(type="text", text="Hello")
        assert block["type"] == "text"
        assert block["text"] == "Hello"

    def test_content_block_invalid_key(self):
        """Test ContentBlock with invalid key."""
        block = ContentBlock(type="text", text="Hello")
        with pytest.raises(KeyError, match="'invalid' is not a valid key"):
            _ = block["invalid"]

    def test_content_block_repr(self):
        """Test ContentBlock string representation."""
        block = ContentBlock(type="text", text="Hello")
        repr_str = repr(block)
        assert "ContentBlock(type='text', text='Hello')" == repr_str


class TestMessage:
    """Test Message class."""

    def test_message_init(self):
        """Test Message initialization."""
        content = [ContentBlock(type="text", text="Hello")]
        usage = Usage(input_tokens=5, output_tokens=10)
        message = Message(
            id="msg_123",
            content=content,
            model="gpt-3.5-turbo",
            role="assistant",
            stop_reason="end_turn",
            usage=usage,
        )
        
        assert message.id == "msg_123"
        assert message.content == content
        assert message.model == "gpt-3.5-turbo"
        assert message.role == "assistant"
        assert message.stop_reason == "end_turn"
        assert message.usage == usage

    def test_message_defaults(self):
        """Test Message with default values."""
        content = [ContentBlock(type="text", text="Hello")]
        message = Message(id="msg_123", content=content, model="gpt-3.5-turbo")
        
        assert message.role == "assistant"
        assert message.stop_reason is None
        assert message.stop_sequence is None
        assert message.type == "message"
        assert message.usage is None

    def test_message_dict_access(self):
        """Test Message dictionary-like access."""
        content = [ContentBlock(type="text", text="Hello")]
        message = Message(id="msg_123", content=content, model="gpt-3.5-turbo")
        
        assert message["id"] == "msg_123"
        assert message["content"] == content
        assert message["model"] == "gpt-3.5-turbo"
        assert message["role"] == "assistant"
        assert message["type"] == "message"

    def test_message_invalid_key(self):
        """Test Message with invalid key."""
        content = [ContentBlock(type="text", text="Hello")]
        message = Message(id="msg_123", content=content, model="gpt-3.5-turbo")
        
        with pytest.raises(KeyError, match="'invalid' is not a valid key"):
            _ = message["invalid"]

    def test_message_repr(self):
        """Test Message string representation."""
        content = [ContentBlock(type="text", text="Hello")]
        message = Message(id="msg_123", content=content, model="gpt-3.5-turbo")
        repr_str = repr(message)
        assert "Message(id='msg_123'" in repr_str
        assert "model='gpt-3.5-turbo'" in repr_str


class TestToolUse:
    """Test ToolUse class."""

    def test_tool_use_init(self):
        """Test ToolUse initialization."""
        tool_use = ToolUse(id="tool_123", name="get_weather", input={"city": "NYC"})
        assert tool_use.id == "tool_123"
        assert tool_use.name == "get_weather"
        assert tool_use.input == {"city": "NYC"}
        assert tool_use.type == "tool_use"

    def test_tool_use_dict_access(self):
        """Test ToolUse dictionary-like access."""
        tool_use = ToolUse(id="tool_123", name="get_weather", input={"city": "NYC"})
        assert tool_use["id"] == "tool_123"
        assert tool_use["name"] == "get_weather"
        assert tool_use["input"] == {"city": "NYC"}
        assert tool_use["type"] == "tool_use"

    def test_tool_use_invalid_key(self):
        """Test ToolUse with invalid key."""
        tool_use = ToolUse(id="tool_123", name="get_weather", input={})
        with pytest.raises(KeyError, match="'invalid' is not a valid key"):
            _ = tool_use["invalid"]

    def test_tool_use_repr(self):
        """Test ToolUse string representation."""
        tool_use = ToolUse(id="tool_123", name="get_weather", input={"city": "NYC"})
        repr_str = repr(tool_use)
        assert "ToolUse(id='tool_123', name='get_weather'" in repr_str


class TestToolResult:
    """Test ToolResult class."""

    def test_tool_result_init(self):
        """Test ToolResult initialization."""
        tool_result = ToolResult(tool_use_id="tool_123", content="Sunny, 72°F")
        assert tool_result.tool_use_id == "tool_123"
        assert tool_result.content == "Sunny, 72°F"
        assert tool_result.is_error is False
        assert tool_result.type == "tool_result"

    def test_tool_result_with_error(self):
        """Test ToolResult with error flag."""
        tool_result = ToolResult(
            tool_use_id="tool_123", content="API error", is_error=True
        )
        assert tool_result.is_error is True

    def test_tool_result_dict_access(self):
        """Test ToolResult dictionary-like access."""
        tool_result = ToolResult(tool_use_id="tool_123", content="Success")
        assert tool_result["tool_use_id"] == "tool_123"
        assert tool_result["content"] == "Success"
        assert tool_result["is_error"] is False
        assert tool_result["type"] == "tool_result"

    def test_tool_result_invalid_key(self):
        """Test ToolResult with invalid key."""
        tool_result = ToolResult(tool_use_id="tool_123", content="Success")
        with pytest.raises(KeyError, match="'invalid' is not a valid key"):
            _ = tool_result["invalid"]

    def test_tool_result_repr(self):
        """Test ToolResult string representation."""
        tool_result = ToolResult(tool_use_id="tool_123", content="Success")
        repr_str = repr(tool_result)
        assert "ToolResult(tool_use_id='tool_123'" in repr_str
        assert "is_error=False" in repr_str


class TestStreamingMessage:
    """Test StreamingMessage class."""

    def test_streaming_message_init(self):
        """Test StreamingMessage initialization."""
        content = [ContentBlock(type="text", text="Hello")]
        usage = Usage(input_tokens=5, output_tokens=10)
        message = StreamingMessage(
            id="msg_123",
            content=content,
            model="gpt-3.5-turbo",
            stop_reason="end_turn",
            usage=usage,
        )
        
        assert message.id == "msg_123"
        assert message.content == content
        assert message.model == "gpt-3.5-turbo"
        assert message.stop_reason == "end_turn"
        assert message.usage == usage

    def test_streaming_message_dict_access(self):
        """Test StreamingMessage dictionary-like access."""
        content = [ContentBlock(type="text", text="Hello")]
        message = StreamingMessage(id="msg_123", content=content, model="gpt-3.5-turbo")
        
        assert message["id"] == "msg_123"
        assert message["content"] == content
        assert message["model"] == "gpt-3.5-turbo"

    def test_streaming_message_invalid_key(self):
        """Test StreamingMessage with invalid key."""
        content = [ContentBlock(type="text", text="Hello")]
        message = StreamingMessage(id="msg_123", content=content, model="gpt-3.5-turbo")
        
        with pytest.raises(KeyError, match="'invalid' is not a valid key"):
            _ = message["invalid"]


class TestMessageDelta:
    """Test MessageDelta class."""

    def test_message_delta_init(self):
        """Test MessageDelta initialization."""
        usage = Usage(5, 10)
        delta = MessageDelta(type="message_delta", delta={"stop_reason": "end_turn"}, usage=usage)
        assert delta.type == "message_delta"
        assert delta.delta == {"stop_reason": "end_turn"}
        assert delta.usage.input_tokens == 5
        assert delta.usage.output_tokens == 10

    def test_message_delta_dict_access(self):
        """Test MessageDelta dictionary-like access."""
        delta = MessageDelta(type="message_delta", delta={"stop_reason": "end_turn"})
        assert delta["type"] == "message_delta"
        assert delta["delta"] == {"stop_reason": "end_turn"}

    def test_message_delta_invalid_key(self):
        """Test MessageDelta with invalid key."""
        delta = MessageDelta()
        with pytest.raises(KeyError, match="'invalid' is not a valid key"):
            _ = delta["invalid"]


class TestContentBlockDelta:
    """Test ContentBlockDelta class."""

    def test_content_block_delta_init(self):
        """Test ContentBlockDelta initialization."""
        delta = ContentBlockDelta(type="content_block_delta", index=0, delta={"text": "Hi"})
        assert delta.type == "content_block_delta"
        assert delta.index == 0
        assert delta.delta == {"text": "Hi"}

    def test_content_block_delta_dict_access(self):
        """Test ContentBlockDelta dictionary-like access."""
        delta = ContentBlockDelta(type="content_block_delta", index=0, delta={"text": "Hi"})
        assert delta["type"] == "content_block_delta"
        assert delta["index"] == 0
        assert delta["delta"] == {"text": "Hi"}

    def test_content_block_delta_invalid_key(self):
        """Test ContentBlockDelta with invalid key."""
        delta = ContentBlockDelta(type="content_block_delta", index=0, delta={})
        with pytest.raises(KeyError, match="'invalid' is not a valid key"):
            _ = delta["invalid"]

    def test_content_block_delta_repr(self):
        """Test ContentBlockDelta string representation."""
        delta = ContentBlockDelta(type="content_block_delta", index=0, delta={"text": "Hi"})
        repr_str = repr(delta)
        assert "ContentBlockDelta(type='content_block_delta', index=0" in repr_str


class TestContentBlockStart:
    """Test ContentBlockStart class."""

    def test_content_block_start_init(self):
        """Test ContentBlockStart initialization."""
        content_block = ContentBlock(type="text", text="")
        start = ContentBlockStart(
            type="content_block_start", index=0, content_block=content_block
        )
        assert start.type == "content_block_start"
        assert start.index == 0
        assert start.content_block == content_block

    def test_content_block_start_dict_access(self):
        """Test ContentBlockStart dictionary-like access."""
        content_block = ContentBlock(type="text", text="")
        start = ContentBlockStart(
            type="content_block_start", index=0, content_block=content_block
        )
        assert start["type"] == "content_block_start"
        assert start["index"] == 0
        assert start["content_block"] == content_block

    def test_content_block_start_invalid_key(self):
        """Test ContentBlockStart with invalid key."""
        content_block = ContentBlock(type="text", text="")
        start = ContentBlockStart(
            type="content_block_start", index=0, content_block=content_block
        )
        with pytest.raises(KeyError, match="'invalid' is not a valid key"):
            _ = start["invalid"]

    def test_content_block_start_repr(self):
        """Test ContentBlockStart string representation."""
        content_block = ContentBlock(type="text", text="")
        start = ContentBlockStart(
            type="content_block_start", index=0, content_block=content_block
        )
        repr_str = repr(start)
        assert "ContentBlockStart(type='content_block_start', index=0" in repr_str


class TestContentBlockStop:
    """Test ContentBlockStop class."""

    def test_content_block_stop_init(self):
        """Test ContentBlockStop initialization."""
        stop = ContentBlockStop(type="content_block_stop", index=0)
        assert stop.type == "content_block_stop"
        assert stop.index == 0

    def test_content_block_stop_dict_access(self):
        """Test ContentBlockStop dictionary-like access."""
        stop = ContentBlockStop(type="content_block_stop", index=0)
        assert stop["type"] == "content_block_stop"
        assert stop["index"] == 0

    def test_content_block_stop_invalid_key(self):
        """Test ContentBlockStop with invalid key."""
        stop = ContentBlockStop(type="content_block_stop", index=0)
        with pytest.raises(KeyError, match="'invalid' is not a valid key"):
            _ = stop["invalid"]


class TestMessageStart:
    """Test MessageStart class."""

    def test_message_start_init(self):
        """Test MessageStart initialization."""
        message = Message(id="msg_123", content=[], model="gpt-3.5-turbo")
        start = MessageStart(type="message_start", message=message)
        assert start.type == "message_start"
        assert start.message == message

    def test_message_start_dict_access(self):
        """Test MessageStart dictionary-like access."""
        message = Message(id="msg_123", content=[], model="gpt-3.5-turbo")
        start = MessageStart(type="message_start", message=message)
        assert start["type"] == "message_start"
        assert start["message"] == message

    def test_message_start_invalid_key(self):
        """Test MessageStart with invalid key."""
        message = Message(id="msg_123", content=[], model="gpt-3.5-turbo")
        start = MessageStart(type="message_start", message=message)
        with pytest.raises(KeyError, match="'invalid' is not a valid key"):
            _ = start["invalid"]


class TestMessageStop:
    """Test MessageStop class."""

    def test_message_stop_init(self):
        """Test MessageStop initialization."""
        stop = MessageStop(type="message_stop")
        assert stop.type == "message_stop"

    def test_message_stop_dict_access(self):
        """Test MessageStop dictionary-like access."""
        stop = MessageStop(type="message_stop")
        assert stop["type"] == "message_stop"

    def test_message_stop_invalid_key(self):
        """Test MessageStop with invalid key."""
        stop = MessageStop(type="message_stop")
        with pytest.raises(KeyError, match="'invalid' is not a valid key"):
            _ = stop["invalid"]