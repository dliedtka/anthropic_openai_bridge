import json
from unittest.mock import Mock, patch

import httpx
import pytest

from anthropic_openai_bridge import AnthropicClient, ToolUse
from anthropic_openai_bridge.transformers.request import (
    _transform_content_blocks, _transform_tool_choice,
    _transform_tools_to_functions)
from anthropic_openai_bridge.transformers.response import \
    transform_openai_to_anthropic


class TestToolTransformation:
    def test_transform_anthropic_tools_to_openai_functions(self):
        anthropic_tools = [
            {
                "name": "get_weather",
                "description": "Get current weather information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            },
        ]

        openai_functions = _transform_tools_to_functions(anthropic_tools)

        assert len(openai_functions) == 2

        # Check first function
        func1 = openai_functions[0]
        assert func1["type"] == "function"
        assert func1["function"]["name"] == "get_weather"
        assert func1["function"]["description"] == "Get current weather information"
        assert "location" in func1["function"]["parameters"]["properties"]
        assert func1["function"]["parameters"]["required"] == ["location"]

        # Check second function
        func2 = openai_functions[1]
        assert func2["type"] == "function"
        assert func2["function"]["name"] == "calculate"

    def test_transform_tool_choice_string_values(self):
        # Test string values
        assert _transform_tool_choice("auto") == "auto"
        assert _transform_tool_choice("any") == "required"
        assert _transform_tool_choice("required") == "required"
        assert _transform_tool_choice("unknown") == "auto"

    def test_transform_tool_choice_specific_tool(self):
        # Test specific tool choice
        anthropic_choice = {"type": "tool", "name": "get_weather"}

        openai_choice = _transform_tool_choice(anthropic_choice)

        assert openai_choice["type"] == "function"
        assert openai_choice["function"]["name"] == "get_weather"

    def test_transform_content_blocks_with_tool_use(self):
        # Test content blocks with tool use
        content_blocks = [
            {"type": "text", "text": "I'll check the weather for you."},
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "San Francisco", "unit": "celsius"},
            },
        ]

        result = _transform_content_blocks("assistant", content_blocks)

        assert result["role"] == "assistant"
        assert result["content"] == "I'll check the weather for you."
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1

        tool_call = result["tool_calls"][0]
        assert tool_call["id"] == "toolu_123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"

        arguments = json.loads(tool_call["function"]["arguments"])
        assert arguments["location"] == "San Francisco"
        assert arguments["unit"] == "celsius"

    def test_transform_content_blocks_with_tool_result(self):
        # Test content blocks with tool results
        content_blocks = [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_123",
                "content": "The weather in San Francisco is sunny, 22°C",
            }
        ]

        result = _transform_content_blocks("user", content_blocks)

        # Should return a list with tool result message
        assert isinstance(result, list)
        assert len(result) == 1

        tool_result = result[0]
        assert tool_result["role"] == "tool"
        assert tool_result["tool_call_id"] == "toolu_123"
        assert tool_result["content"] == "The weather in San Francisco is sunny, 22°C"

    def test_transform_content_blocks_mixed(self):
        # Test mixed content: text + tool_use
        content_blocks = [
            {"type": "text", "text": "Let me calculate that for you."},
            {
                "type": "tool_use",
                "id": "toolu_456",
                "name": "calculate",
                "input": {"expression": "2 + 2"},
            },
        ]

        result = _transform_content_blocks("assistant", content_blocks)

        assert result["role"] == "assistant"
        assert result["content"] == "Let me calculate that for you."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "calculate"


class TestToolResponseTransformation:
    def test_transform_openai_tool_response_to_anthropic(self):
        # Test OpenAI response with tool calls
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
                        "content": "I'll check the weather for you.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40},
        }

        anthropic_response = transform_openai_to_anthropic(openai_response)

        assert anthropic_response.id == "chatcmpl-123"
        assert anthropic_response.role == "assistant"
        assert anthropic_response.stop_reason == "tool_use"
        assert len(anthropic_response.content) == 2  # text + tool_use

        # Check text content
        text_content = anthropic_response.content[0]
        assert text_content.type == "text"
        assert text_content.text == "I'll check the weather for you."

        # Check tool use content
        tool_content = anthropic_response.content[1]
        assert isinstance(tool_content, ToolUse)
        assert tool_content.id == "call_123"
        assert tool_content.name == "get_weather"
        assert tool_content.input["location"] == "San Francisco"
        assert tool_content.input["unit"] == "celsius"

        # Check usage
        assert anthropic_response.usage.input_tokens == 25
        assert anthropic_response.usage.output_tokens == 15


class TestToolIntegration:
    @patch("anthropic_openai_bridge.messages.httpx.Client.post")
    def test_tool_calling_request_transformation(self, mock_post):
        # Mock successful tool call response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Boston"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")

        response = client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            messages=[{"role": "user", "content": "What's the weather in Boston?"}],
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
            tool_choice="auto",
        )

        # Verify the request was transformed correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        request_data = call_args[1]["json"]

        # Check that tools were transformed to OpenAI format
        assert "tools" in request_data
        assert len(request_data["tools"]) == 1
        assert request_data["tools"][0]["type"] == "function"
        assert request_data["tools"][0]["function"]["name"] == "get_weather"

        # Check tool_choice was transformed
        assert request_data["tool_choice"] == "auto"

        # Verify response format
        assert isinstance(response.content[0], ToolUse)
        assert response.content[0].name == "get_weather"
        assert response.content[0].input["location"] == "Boston"

    @patch("anthropic_openai_bridge.messages.httpx.Client.post")
    def test_tool_conversation_flow(self, mock_post):
        # Test a complete tool conversation: user -> assistant (tool call) -> user (tool result) -> assistant
        client = AnthropicClient(api_key="test-key")

        # First response: assistant makes tool call
        mock_response_1 = Mock(spec=httpx.Response)
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
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
                                    "arguments": '{"location": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35},
        }

        # Second response: assistant responds with weather info
        mock_response_2 = Mock(spec=httpx.Response)
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            "id": "chatcmpl-456",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The weather in Paris is sunny with 25°C.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        }

        mock_post.side_effect = [mock_response_1, mock_response_2]

        # First call: user asks about weather
        response_1 = client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        )

        # Verify first response has tool call
        assert len(response_1.content) == 2
        assert response_1.content[0].text == "I'll check the weather for you."
        assert isinstance(response_1.content[1], ToolUse)
        assert response_1.content[1].name == "get_weather"

        # Second call: include tool result
        conversation = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll check the weather for you."},
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "get_weather",
                        "input": {"location": "Paris"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "Sunny, 25°C",
                    }
                ],
            },
        ]

        response_2 = client.messages.create(
            model="gpt-3.5-turbo", max_tokens=1000, messages=conversation
        )

        # Verify second response
        assert response_2.content[0].text == "The weather in Paris is sunny with 25°C."

        # Verify the conversation was transformed correctly in the request
        assert mock_post.call_count == 2
        second_call_args = mock_post.call_args_list[1]
        request_messages = second_call_args[1]["json"]["messages"]

        # Should have: user, assistant (with tool_calls), tool (result), assistant
        assert len(request_messages) >= 3

        # Check tool result was transformed correctly
        tool_message_found = False
        for msg in request_messages:
            if msg.get("role") == "tool":
                tool_message_found = True
                assert msg["tool_call_id"] == "call_123"
                assert msg["content"] == "Sunny, 25°C"

        assert (
            tool_message_found
        ), "Tool result message should be present in the request"

    @patch("anthropic_openai_bridge.messages.httpx.Client.post")
    def test_tool_choice_variations(self, mock_post):
        # Test different tool_choice values
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="test-key")
        tools = [
            {
                "name": "test_tool",
                "description": "Test",
                "input_schema": {"type": "object"},
            }
        ]

        # Test auto choice
        client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=100,
            messages=[{"role": "user", "content": "test"}],
            tools=tools,
            tool_choice="auto",
        )

        request_data = mock_post.call_args[1]["json"]
        assert request_data["tool_choice"] == "auto"

        # Test required choice
        client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=100,
            messages=[{"role": "user", "content": "test"}],
            tools=tools,
            tool_choice="required",
        )

        request_data = mock_post.call_args[1]["json"]
        assert request_data["tool_choice"] == "required"

        # Test specific tool choice
        client.messages.create(
            model="gpt-3.5-turbo",
            max_tokens=100,
            messages=[{"role": "user", "content": "test"}],
            tools=tools,
            tool_choice={"type": "tool", "name": "test_tool"},
        )

        request_data = mock_post.call_args[1]["json"]
        assert request_data["tool_choice"]["type"] == "function"
        assert request_data["tool_choice"]["function"]["name"] == "test_tool"
