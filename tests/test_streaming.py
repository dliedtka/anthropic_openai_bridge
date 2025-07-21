from unittest.mock import Mock, patch

import httpx
import pytest

from anthropic_openai_bridge import AnthropicClient
from anthropic_openai_bridge.streaming import (
    SSEParser, parse_openai_streaming_response,
    transform_openai_stream_to_anthropic)
from anthropic_openai_bridge.types import (ContentBlockDelta,
                                           ContentBlockStart, ContentBlockStop,
                                           MessageDelta, MessageStart,
                                           MessageStop)


class TestSSEParser:
    def test_parse_line(self):
        # Test event line
        result = SSEParser.parse_line("event: message_start")
        assert result == {"event": "message_start"}

        # Test data line
        result = SSEParser.parse_line('data: {"id": "msg_123"}')
        assert result == {"data": '{"id": "msg_123"}'}

        # Test comment line (should be ignored)
        result = SSEParser.parse_line(": this is a comment")
        assert result is None

        # Test empty line
        result = SSEParser.parse_line("")
        assert result is None

    def test_parse_event(self):
        event_data = """event: message_start
data: {"id": "msg_123", "type": "message"}"""

        result = SSEParser.parse_event(event_data)
        assert result["event"] == "message_start"
        assert result["data"]["id"] == "msg_123"
        assert result["data"]["type"] == "message"

    def test_parse_done_event(self):
        event_data = "data: [DONE]"
        result = SSEParser.parse_event(event_data)
        assert result["event"] == "done"


class TestStreamingResponse:
    @patch("anthropic_openai_bridge.messages.httpx.Client.post")
    def test_streaming_text_response(self, mock_post):
        # Mock streaming response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.iter_text.return_value = [
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Hello"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" world"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
            "data: [DONE]\n\n",
        ]
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")

        stream = client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        events = list(stream)

        # Should have message_start, content_block_start, content_block_delta(s),
        # content_block_stop, message_delta, message_stop
        assert len(events) >= 5

        # First event should be message start
        assert isinstance(events[0], MessageStart)
        assert events[0].message.id == "chatcmpl-123"

        # Should have content block events
        content_block_start_found = False
        content_block_delta_found = False
        content_block_stop_found = False
        message_stop_found = False

        for event in events:
            if isinstance(event, ContentBlockStart):
                content_block_start_found = True
            elif isinstance(event, ContentBlockDelta):
                content_block_delta_found = True
                assert "text" in event.delta
            elif isinstance(event, ContentBlockStop):
                content_block_stop_found = True
            elif isinstance(event, MessageStop):
                message_stop_found = True

        assert content_block_start_found
        assert content_block_delta_found
        assert content_block_stop_found
        assert message_stop_found

    @patch("anthropic_openai_bridge.messages.httpx.Client.post")
    def test_streaming_tool_response(self, mock_post):
        # Mock streaming response with tool calls
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.iter_text.return_value = [
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"tool_calls":[{"id":"call_123","function":{"name":"get_weather","arguments":"{\\"location\\": \\"San Francisco\\"}"}}]}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":15,"completion_tokens":10}}\n\n',
            "data: [DONE]\n\n",
        ]
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")

        stream = client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
            stream=True,
        )

        events = list(stream)

        # Should have message_start, content_block_start (for tool),
        # content_block_delta, content_block_stop, message_delta, message_stop
        assert len(events) >= 4

        # First event should be message start
        assert isinstance(events[0], MessageStart)

        # Should have tool-related events
        tool_found = False
        for event in events:
            if isinstance(event, ContentBlockStart):
                if hasattr(event.content_block, "name"):
                    tool_found = True
                    assert event.content_block.name == "get_weather"
            elif isinstance(event, MessageDelta):
                if event.delta.get("stop_reason") == "tool_use":
                    assert True  # Correct stop reason mapping

        assert tool_found


class TestStreamingTransformation:
    def test_transform_openai_to_anthropic_text(self):
        # Test transforming OpenAI streaming events to Anthropic format
        openai_events = [
            {"id": "chatcmpl-123", "choices": [{"delta": {"role": "assistant"}}]},
            {"id": "chatcmpl-123", "choices": [{"delta": {"content": "Hello"}}]},
            {"id": "chatcmpl-123", "choices": [{"delta": {"content": " world"}}]},
            {
                "id": "chatcmpl-123",
                "choices": [{"finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        ]

        anthropic_events = list(
            transform_openai_stream_to_anthropic(iter(openai_events))
        )

        # Should have message_start, content_block_start, 2 content_block_deltas,
        # content_block_stop, message_delta, message_stop
        assert len(anthropic_events) >= 6

        # Check event types and content
        assert isinstance(anthropic_events[0], MessageStart)
        assert isinstance(anthropic_events[1], ContentBlockStart)
        assert isinstance(anthropic_events[2], ContentBlockDelta)
        assert anthropic_events[2].delta["text"] == "Hello"
        assert isinstance(anthropic_events[3], ContentBlockDelta)
        assert anthropic_events[3].delta["text"] == " world"

    def test_transform_finish_reasons(self):
        test_cases = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("function_call", "tool_use"),
            ("content_filter", "end_turn"),
        ]

        for openai_reason, expected_anthropic_reason in test_cases:
            openai_events = [
                {"id": "test", "choices": [{"delta": {"role": "assistant"}}]},
                {
                    "id": "test",
                    "choices": [{"finish_reason": openai_reason}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ]

            anthropic_events = list(
                transform_openai_stream_to_anthropic(iter(openai_events))
            )

            # Find the MessageDelta event
            message_delta = None
            for event in anthropic_events:
                if isinstance(event, MessageDelta):
                    message_delta = event
                    break

            assert message_delta is not None
            assert message_delta.delta["stop_reason"] == expected_anthropic_reason


class TestStreamingIntegration:
    @patch("anthropic_openai_bridge.messages.httpx.Client.post")
    def test_end_to_end_streaming(self, mock_post):
        # Test complete streaming workflow
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.iter_text.return_value = [
            'data: {"id":"chatcmpl-123","model":"gpt-3.5-turbo","choices":[{"delta":{"role":"assistant"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"The"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" weather"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" is"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" sunny."}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":8}}\n\n',
            "data: [DONE]\n\n",
        ]
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")

        stream = client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            messages=[{"role": "user", "content": "What's the weather like?"}],
            stream=True,
        )

        # Collect all events
        events = list(stream)

        # Verify we get the expected sequence
        message_start = events[0]
        assert isinstance(message_start, MessageStart)
        assert message_start.message.id == "chatcmpl-123"
        assert message_start.message.model == "gpt-3.5-turbo"

        # Verify final message has complete content
        complete_text = ""
        for event in events:
            if isinstance(event, ContentBlockDelta) and "text" in event.delta:
                complete_text += event.delta["text"]

        assert complete_text == "The weather is sunny."

        # Verify usage information is included
        message_delta_events = [e for e in events if isinstance(e, MessageDelta)]
        assert len(message_delta_events) > 0
        final_delta = message_delta_events[-1]
        assert final_delta.usage is not None
        assert final_delta.usage.input_tokens == 20
        assert final_delta.usage.output_tokens == 8
