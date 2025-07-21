import pytest

from anthropic_openai_bridge.transformers.request import \
    transform_anthropic_to_openai
from anthropic_openai_bridge.transformers.response import \
    transform_openai_to_anthropic


class TestRequestTransformer:
    def test_basic_message_transformation(self):
        anthropic_params = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "max_tokens": 1000,
        }

        openai_params = transform_anthropic_to_openai(anthropic_params)

        assert openai_params["model"] == "claude-3-sonnet-20240229"
        assert openai_params["max_tokens"] == 1000
        assert len(openai_params["messages"]) == 1
        assert openai_params["messages"][0]["role"] == "user"
        assert openai_params["messages"][0]["content"] == "Hello, world!"

    def test_system_message_transformation(self):
        anthropic_params = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "max_tokens": 1000,
            "system": "You are a helpful assistant.",
        }

        openai_params = transform_anthropic_to_openai(anthropic_params)

        assert len(openai_params["messages"]) == 2
        assert openai_params["messages"][0]["role"] == "system"
        assert openai_params["messages"][0]["content"] == "You are a helpful assistant."
        assert openai_params["messages"][1]["role"] == "user"
        assert openai_params["messages"][1]["content"] == "Hello, world!"

    def test_optional_parameters(self):
        anthropic_params = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop_sequences": ["STOP"],
        }

        openai_params = transform_anthropic_to_openai(anthropic_params)

        assert openai_params["temperature"] == 0.7
        assert openai_params["top_p"] == 0.9
        assert openai_params["stop"] == ["STOP"]

    def test_content_blocks(self):
        anthropic_params = {
            "model": "claude-3-sonnet-20240229",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello, world!"},
                        {"type": "text", "text": "How are you?"},
                    ],
                }
            ],
            "max_tokens": 1000,
        }

        openai_params = transform_anthropic_to_openai(anthropic_params)

        assert openai_params["messages"][0]["content"] == "Hello, world!\nHow are you?"


class TestResponseTransformer:
    def test_basic_response_transformation(self):
        openai_response = {
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

        anthropic_response = transform_openai_to_anthropic(openai_response)

        assert anthropic_response.id == "chatcmpl-123"
        assert anthropic_response.type == "message"
        assert anthropic_response.role == "assistant"
        assert anthropic_response.model == "gpt-3.5-turbo"
        assert anthropic_response.stop_reason == "end_turn"
        assert len(anthropic_response.content) == 1
        assert anthropic_response.content[0].type == "text"
        assert anthropic_response.content[0].text == "Hello! How can I help you today?"
        assert anthropic_response.usage.input_tokens == 9
        assert anthropic_response.usage.output_tokens == 12

        # Test dict-like access for backward compatibility
        assert anthropic_response["id"] == "chatcmpl-123"
        assert anthropic_response["type"] == "message"
        assert anthropic_response["role"] == "assistant"
        assert anthropic_response["model"] == "gpt-3.5-turbo"
        assert anthropic_response["stop_reason"] == "end_turn"
        assert len(anthropic_response["content"]) == 1
        assert anthropic_response["content"][0]["type"] == "text"
        assert (
            anthropic_response["content"][0]["text"]
            == "Hello! How can I help you today?"
        )
        assert anthropic_response["usage"]["input_tokens"] == 9
        assert anthropic_response["usage"]["output_tokens"] == 12

    def test_finish_reason_mapping(self):
        test_cases = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("function_call", "tool_use"),
            ("tool_calls", "tool_use"),
            ("content_filter", "end_turn"),
            (None, None),
        ]

        for openai_reason, expected_anthropic_reason in test_cases:
            openai_response = {
                "id": "test",
                "choices": [
                    {"finish_reason": openai_reason, "message": {"content": "test"}}
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }

            anthropic_response = transform_openai_to_anthropic(openai_response)
            assert anthropic_response.stop_reason == expected_anthropic_reason
