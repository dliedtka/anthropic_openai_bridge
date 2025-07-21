import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from anthropic_openai_bridge import AsyncAnthropicClient
from anthropic_openai_bridge.types import (ContentBlockDelta, MessageStart,
                                           MessageStop)


class TestAsyncClient:
    @pytest.mark.asyncio
    async def test_async_client_initialization(self):
        client = AsyncAnthropicClient(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        assert client.api_key == "test-key"
        assert client.base_url == "https://api.test.com/v1"
        assert client.messages is not None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with AsyncAnthropicClient(api_key="test-key") as client:
            assert client is not None

    @pytest.mark.asyncio
    @patch("anthropic_openai_bridge.async_messages.httpx.AsyncClient.post")
    async def test_async_message_creation(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }

        # Configure mock to return the response directly
        mock_post.return_value = mock_response

        client = AsyncAnthropicClient(api_key="test-key")

        response = await client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello, world!"}],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert response.content[0].text == "Hello! How can I help you today?"

        # Test dict-like access
        assert response["type"] == "message"
        assert response["content"][0]["text"] == "Hello! How can I help you today?"

        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch("anthropic_openai_bridge.async_messages.httpx.AsyncClient.post")
    async def test_async_streaming(self, mock_post):
        # Mock async streaming response
        mock_response = Mock()
        mock_response.status_code = 200

        # Create an async iterator for aiter_text
        async def mock_aiter_text():
            chunks = [
                'data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant"}}]}\n\n',
                'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Hello"}}]}\n\n',
                'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" world"}}]}\n\n',
                'data: {"id":"chatcmpl-123","choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
                "data: [DONE]\n\n",
            ]
            for chunk in chunks:
                yield chunk

        mock_response.aiter_text.return_value = mock_aiter_text()

        # Configure mock to return the response directly
        mock_post.return_value = mock_response

        client = AsyncAnthropicClient(api_key="test-key")

        stream = await client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        events = []
        async for event in stream:
            events.append(event)

        # Should have message_start, content_block_start, content_block_deltas, etc.
        assert len(events) >= 5

        # First event should be message start
        assert isinstance(events[0], MessageStart)
        assert events[0].message.id == "chatcmpl-123"

        # Should have content deltas
        content_deltas = [e for e in events if isinstance(e, ContentBlockDelta)]
        assert len(content_deltas) >= 2

        # Last event should be message stop
        assert isinstance(events[-1], MessageStop)

    @pytest.mark.asyncio
    @patch("anthropic_openai_bridge.async_messages.httpx.AsyncClient.post")
    async def test_async_tool_calling(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'll check the weather for you.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "New York"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        }

        # Configure mock to return the response directly
        mock_post.return_value = mock_response

        client = AsyncAnthropicClient(api_key="test-key")

        response = await client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
        )

        # Verify response format
        assert len(response.content) == 2  # text + tool_use
        assert response.content[0].text == "I'll check the weather for you."
        assert response.content[1].name == "get_weather"
        assert response.content[1].input["location"] == "New York"
        assert response.stop_reason == "tool_use"

        mock_post.assert_called_once()

        # Verify request transformation
        call_args = mock_post.call_args
        request_data = call_args[1]["json"]
        assert "tools" in request_data
        assert request_data["tools"][0]["function"]["name"] == "get_weather"


class TestAsyncStreamingIntegration:
    @pytest.mark.asyncio
    @patch("anthropic_openai_bridge.async_messages.httpx.AsyncClient.post")
    async def test_async_streaming_tool_calls(self, mock_post):
        # Mock async streaming response with tool calls
        mock_response = Mock()
        mock_response.status_code = 200

        async def mock_aiter_text():
            chunks = [
                'data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant"}}]}\n\n',
                'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Let me check that for you."}}]}\n\n',
                'data: {"id":"chatcmpl-123","choices":[{"delta":{"tool_calls":[{"id":"call_123","function":{"name":"search","arguments":"{\\"query\\": \\"Python async\\"}"}}]}}]}\n\n',
                'data: {"id":"chatcmpl-123","choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":25,"completion_tokens":20}}\n\n',
                "data: [DONE]\n\n",
            ]
            for chunk in chunks:
                yield chunk

        mock_response.aiter_text.return_value = mock_aiter_text()
        # Configure mock to return the response directly
        mock_post.return_value = mock_response

        client = AsyncAnthropicClient(api_key="test-key")

        stream = await client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Search for Python async tutorials"}],
            tools=[
                {
                    "name": "search",
                    "description": "Search the web",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ],
            stream=True,
        )

        events = []
        async for event in stream:
            events.append(event)

        # Verify we get appropriate events for tool call streaming
        assert len(events) >= 4

        # Should have message start
        message_starts = [e for e in events if isinstance(e, MessageStart)]
        assert len(message_starts) == 1

        # Should have content deltas
        content_deltas = [e for e in events if isinstance(e, ContentBlockDelta)]
        assert len(content_deltas) >= 1

        # Check for tool use in content deltas
        tool_found = False
        for event in events:
            if hasattr(event, "delta") and "input" in event.delta:
                tool_found = True

        # Message should end with tool_use stop reason
        message_stops = [e for e in events if isinstance(e, MessageStop)]
        assert len(message_stops) == 1

    @pytest.mark.asyncio
    @patch("anthropic_openai_bridge.async_messages.httpx.AsyncClient.post")
    async def test_async_error_handling(self, mock_post):
        from anthropic_openai_bridge import AuthenticationError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key", "type": "invalid_request_error"}
        }

        # Configure mock to return the response directly
        mock_post.return_value = mock_response

        client = AsyncAnthropicClient(api_key="invalid-key")

        with pytest.raises(AuthenticationError) as exc_info:
            await client.messages.create(
                model="gpt-3.5-turbo",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_context_manager_cleanup(self):
        # Test that HTTP client is properly closed
        mock_http_client = Mock()
        mock_http_client.aclose = AsyncMock()

        async with AsyncAnthropicClient(
            api_key="test-key", http_client=mock_http_client
        ) as client:
            pass

        mock_http_client.aclose.assert_called_once()


class TestAsyncPerformance:
    @pytest.mark.asyncio
    @patch("anthropic_openai_bridge.async_messages.httpx.AsyncClient.post")
    async def test_concurrent_requests(self, mock_post):
        # Test multiple concurrent async requests
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        # Configure mock to return the response directly
        mock_post.return_value = mock_response

        client = AsyncAnthropicClient(api_key="test-key")

        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            task = client.messages.create(
                model="gpt-3.5-turbo",
                max_tokens=100,
                messages=[{"role": "user", "content": f"Request {i}"}],
            )
            tasks.append(task)

        # Wait for all to complete
        responses = await asyncio.gather(*tasks)

        # Verify all responses
        assert len(responses) == 5
        for response in responses:
            assert response.content[0].text == "Response"

        # Verify all requests were made
        assert mock_post.call_count == 5
