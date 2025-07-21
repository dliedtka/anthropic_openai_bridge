from unittest.mock import Mock, patch

import httpx
import pytest

from anthropic_openai_bridge import AnthropicClient
from anthropic_openai_bridge.exceptions import (AuthenticationError,
                                                RateLimitError)


class TestAnthropicClient:
    def test_client_initialization(self):
        client = AnthropicClient(api_key="test-key", base_url="https://api.test.com/v1")

        assert client.api_key == "test-key"
        assert client.base_url == "https://api.test.com/v1"
        assert client.messages is not None

    def test_context_manager(self):
        with AnthropicClient(api_key="test-key") as client:
            assert client is not None

    @patch("anthropic_openai_bridge.messages.httpx.Client.post")
    def test_successful_message_creation(self, mock_post):
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
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello, world!"}],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert response.content[0].text == "Hello! How can I help you today?"

        # Test dict-like access for backward compatibility
        assert response["type"] == "message"
        assert response["role"] == "assistant"
        assert len(response["content"]) == 1
        assert response["content"][0]["text"] == "Hello! How can I help you today?"

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "claude-3-sonnet-20240229"
        assert call_args[1]["json"]["max_tokens"] == 1000

    @patch("anthropic_openai_bridge.messages.httpx.Client.post")
    def test_authentication_error(self, mock_post):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key", "type": "invalid_request_error"}
        }
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="invalid-key")

        with pytest.raises(AuthenticationError) as exc_info:
            client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert "Invalid API key" in str(exc_info.value)

    @patch("anthropic_openai_bridge.messages.httpx.Client.post")
    def test_rate_limit_error(self, mock_post):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
        }
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")

        with pytest.raises(RateLimitError) as exc_info:
            client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert "Rate limit exceeded" in str(exc_info.value)

    @patch("anthropic_openai_bridge.messages.httpx.Client.post")
    def test_streaming_and_tools_supported(self, mock_post):
        # Test that streaming and tools are now supported (no NotImplementedError)
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.iter_text.return_value = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")

        # Test streaming - should not raise NotImplementedError
        try:
            result = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            # Should return an iterator for streaming
            assert hasattr(result, "__iter__")
        except NotImplementedError:
            pytest.fail("Streaming should be implemented in Phase 2")

        # Reset mock for tools test
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
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "test_tool",
                                    "arguments": '{"arg": "value"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
        mock_post.return_value = mock_response

        # Test tools - should not raise NotImplementedError
        try:
            result = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Hello"}],
                tools=[
                    {
                        "name": "test_tool",
                        "description": "Test tool",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
            )
            # Should return a Message object
            assert hasattr(result, "content")
        except NotImplementedError:
            pytest.fail("Tool calling should be implemented in Phase 2")
