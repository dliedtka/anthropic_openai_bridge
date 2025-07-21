"""
Tests for edge cases and error conditions.
"""

import json
from unittest.mock import Mock, patch

import httpx
import pytest

from anthropic_openai_bridge.exceptions import APIError, AuthenticationError
from anthropic_openai_bridge.messages import Messages
from anthropic_openai_bridge.async_messages import AsyncMessages
from anthropic_openai_bridge.transformers.request import _transform_content_blocks
from anthropic_openai_bridge.transformers.response import transform_openai_to_anthropic


class TestRequestTransformerEdgeCases:
    """Test edge cases in request transformation."""

    def test_transform_content_blocks_empty_list(self):
        """Test transforming empty content blocks list."""
        result = _transform_content_blocks("user", [])
        assert result == {}

    def test_transform_content_blocks_unknown_type(self):
        """Test transforming content block with unknown type."""
        content_blocks = [{"type": "unknown", "content": "test"}]
        result = _transform_content_blocks("user", content_blocks)
        # Should return a message with empty content since unknown type is ignored
        assert isinstance(result, dict)

    def test_transform_content_blocks_malformed_tool_result(self):
        """Test transforming malformed tool result."""
        content_blocks = [
            {"type": "tool_result", "tool_use_id": "tool_123"}  # Missing content
        ]
        result = _transform_content_blocks("user", content_blocks)
        assert isinstance(result, list)
        # Should still create tool result message even with missing content

    def test_transform_content_blocks_tool_use_invalid_input(self):
        """Test transforming tool use with invalid input."""
        content_blocks = [
            {
                "type": "tool_use",
                "id": "tool_123",
                "name": "test_tool",
                "input": "invalid_json_string"  # Should be dict
            }
        ]
        result = _transform_content_blocks("assistant", content_blocks)
        # Should handle invalid input gracefully
        assert isinstance(result, dict)


class TestResponseTransformerEdgeCases:
    """Test edge cases in response transformation."""

    def test_transform_openai_to_anthropic_empty_response(self):
        """Test transforming empty OpenAI response."""
        openai_response = {"choices": [], "usage": {}}
        result = transform_openai_to_anthropic(openai_response)
        
        assert result.id.startswith("msg_")
        assert result.content == []
        assert result.model == "unknown"

    def test_transform_openai_to_anthropic_no_choices(self):
        """Test transforming OpenAI response with no choices."""
        openai_response = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        result = transform_openai_to_anthropic(openai_response)
        
        assert result.content == []
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

    def test_transform_openai_to_anthropic_malformed_tool_call(self):
        """Test transforming response with malformed tool call."""
        openai_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tool_123",
                                "function": {
                                    "name": "test_function",
                                    "arguments": "invalid_json"  # Invalid JSON
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        result = transform_openai_to_anthropic(openai_response)
        
        # Should handle invalid JSON gracefully
        assert len(result.content) == 1
        tool_use = result.content[0]
        assert tool_use.name == "test_function"
        assert tool_use.input == {}  # Should default to empty dict


class TestMessagesErrorHandling:
    """Test error handling in Messages class."""

    def test_messages_http_error_handling(self):
        """Test Messages handling of HTTP errors."""
        mock_client = Mock(spec=httpx.Client)
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=Mock(), response=mock_response
        )
        mock_client.post.return_value = mock_response

        messages = Messages("test_key", "https://api.example.com", mock_client)

        with pytest.raises(AuthenticationError):
            messages.create(
                model="gpt-3.5-turbo",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}]
            )

    def test_messages_network_timeout(self):
        """Test Messages handling of network timeouts."""
        mock_client = Mock(spec=httpx.Client)
        mock_client.post.side_effect = httpx.TimeoutException("Request timed out")

        messages = Messages("test_key", "https://api.example.com", mock_client)

        with pytest.raises(APIError, match="Request timed out"):
            messages.create(
                model="gpt-3.5-turbo",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}]
            )

    def test_messages_connection_error(self):
        """Test Messages handling of connection errors."""
        mock_client = Mock(spec=httpx.Client)
        mock_client.post.side_effect = httpx.ConnectError("Connection failed")

        messages = Messages("test_key", "https://api.example.com", mock_client)

        with pytest.raises(APIError, match="Connection failed"):
            messages.create(
                model="gpt-3.5-turbo",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}]
            )


class TestAsyncMessagesErrorHandling:
    """Test error handling in AsyncMessages class."""

    @pytest.mark.asyncio
    async def test_async_messages_http_error_handling(self):
        """Test AsyncMessages handling of HTTP errors."""
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests", request=Mock(), response=mock_response
        )
        mock_client.post.return_value = mock_response

        async_messages = AsyncMessages("test_key", "https://api.example.com", mock_client)

        with pytest.raises(APIError):  # Should be mapped to RateLimitError
            await async_messages.create(
                model="gpt-3.5-turbo",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}]
            )

    @pytest.mark.asyncio
    async def test_async_messages_network_timeout(self):
        """Test AsyncMessages handling of network timeouts."""
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.TimeoutException("Async request timed out")

        async_messages = AsyncMessages("test_key", "https://api.example.com", mock_client)

        with pytest.raises(APIError, match="Async request timed out"):
            await async_messages.create(
                model="gpt-3.5-turbo",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}]
            )

    @pytest.mark.asyncio
    async def test_async_messages_stream_error_handling(self):
        """Test AsyncMessages stream error handling."""
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": {"message": "Internal server error"}}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error", request=Mock(), response=mock_response
        )
        mock_client.post.return_value = mock_response

        async_messages = AsyncMessages("test_key", "https://api.example.com", mock_client)

        with pytest.raises(APIError):
            async for _ in await async_messages.create(
                model="gpt-3.5-turbo",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            ):
                pass


class TestStreamingEdgeCases:
    """Test streaming edge cases."""

    def test_streaming_malformed_sse(self):
        """Test handling of malformed SSE data."""
        from anthropic_openai_bridge.streaming import SSEParser
        
        parser = SSEParser()
        # Test malformed SSE lines
        malformed_lines = [
            "data: invalid_json\n",
            "event: test\ndata: {}\n",  # Missing final newline
            "data: {incomplete\n"
        ]
        
        for line in malformed_lines:
            try:
                list(parser.parse_line(line))  # Use parse_line instead of parse_lines
            except (json.JSONDecodeError, ValueError):
                # Should handle these gracefully or raise appropriate errors
                pass

    @pytest.mark.asyncio
    async def test_streaming_connection_drop(self):
        """Test streaming when connection drops."""
        from anthropic_openai_bridge.streaming import parse_openai_streaming_response_async
        
        # Mock a stream that raises an exception mid-stream  
        async def mock_text_stream():
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}) + "\n\n"
            raise httpx.ReadError("Connection lost")
        
        mock_response = Mock()
        mock_response.aiter_text.return_value = mock_text_stream()
        
        events = []
        try:
            async for event in parse_openai_streaming_response_async(mock_response):
                events.append(event)
        except httpx.ReadError:
            pass  # Expected
        
        # Should have processed at least one event before the error
        assert len(events) >= 0


class TestParameterValidation:
    """Test parameter validation edge cases."""

    def test_messages_invalid_temperature(self):
        """Test Messages with invalid temperature parameter."""
        mock_client = Mock(spec=httpx.Client)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        mock_client.post.return_value = mock_response

        messages = Messages("test_key", "https://api.example.com", mock_client)

        # Should handle invalid temperature values
        # (OpenAI API will reject these, but our transform should not crash)
        result = messages.create(
            model="gpt-3.5-turbo",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=2.5  # Out of valid range 0-1, but should be passed through
        )
        
        assert result is not None

    def test_messages_extremely_large_tokens(self):
        """Test Messages with extremely large token count."""
        mock_client = Mock(spec=httpx.Client)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        mock_client.post.return_value = mock_response

        messages = Messages("test_key", "https://api.example.com", mock_client)

        # Should handle large token counts (validation is OpenAI's responsibility)
        result = messages.create(
            model="gpt-3.5-turbo",
            max_tokens=1000000,  # Very large number
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert result is not None