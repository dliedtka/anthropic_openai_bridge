import json
import logging
from typing import Any, Dict, List, Optional, Union

from ..types import AnthropicCreateParams, OpenAICreateParams

logger = logging.getLogger(__name__)


def transform_anthropic_to_openai(params: AnthropicCreateParams) -> OpenAICreateParams:
    """
    Transform Anthropic Messages API parameters to OpenAI ChatCompletion format.

    This function converts request parameters from Anthropic's Messages API format
    to OpenAI's ChatCompletion API format, handling all supported parameters and
    special cases like system messages, tools, and content blocks.

    Args:
        params: Dictionary containing Anthropic Messages API parameters.
                Must include 'model' and 'messages' keys. Optional keys include
                'system', 'max_tokens', 'temperature', 'top_p', 'stop_sequences',
                'stream', 'tools', and 'tool_choice'.

    Returns:
        Dictionary formatted for OpenAI ChatCompletion API with keys like
        'model', 'messages', 'max_tokens', 'temperature', 'top_p', 'stop',
        'stream', 'functions', and 'function_call'.

    Example:
        >>> anthropic_params = {
        ...     "model": "claude-3-sonnet-20240229",
        ...     "max_tokens": 1000,
        ...     "messages": [{"role": "user", "content": "Hello"}],
        ...     "system": "You are helpful",
        ...     "temperature": 0.7
        ... }
        >>> openai_params = transform_anthropic_to_openai(anthropic_params)
        >>> openai_params["model"]
        'claude-3-sonnet-20240229'
    """
    openai_params: OpenAICreateParams = {}

    openai_params["model"] = params["model"]

    messages = _transform_messages(params["messages"], params.get("system"))
    openai_params["messages"] = messages

    if "max_tokens" in params:
        openai_params["max_tokens"] = params["max_tokens"]

    if "temperature" in params:
        openai_params["temperature"] = params["temperature"]

    if "top_p" in params:
        openai_params["top_p"] = params["top_p"]

    if "stop_sequences" in params:
        openai_params["stop"] = params["stop_sequences"]

    if "stream" in params:
        openai_params["stream"] = params["stream"]

    # Transform tools to OpenAI function format
    if "tools" in params and params["tools"]:
        openai_params["tools"] = _transform_tools_to_functions(params["tools"])

    # Transform tool_choice
    if "tool_choice" in params and params["tool_choice"]:
        openai_params["tool_choice"] = _transform_tool_choice(params["tool_choice"])

    logger.debug(f"Transformed Anthropic params to OpenAI: {openai_params}")
    return openai_params


def _transform_messages(
    anthropic_messages: List[Dict[str, Any]], system_prompt: Optional[str] = None
) -> List[Dict[str, Any]]:
    openai_messages = []

    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    for msg in anthropic_messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Handle complex content blocks (text, tool_use, tool_result)
            transformed_msg = _transform_content_blocks(role, content)
            if transformed_msg:
                if isinstance(transformed_msg, list):
                    openai_messages.extend(transformed_msg)
                else:
                    openai_messages.append(transformed_msg)

    return openai_messages


def _transform_content_blocks(
    role: str, content_blocks: List[Dict[str, Any]]
) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
    """Transform Anthropic content blocks to OpenAI message format."""
    text_parts = []
    tool_calls = []
    tool_results = []

    for block in content_blocks:
        block_type = block.get("type")

        if block_type == "text" and "text" in block:
            text_parts.append(block["text"])

        elif block_type == "tool_use":
            # Convert Anthropic tool_use to OpenAI function call format
            tool_call = {
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            }
            tool_calls.append(tool_call)

        elif block_type == "tool_result":
            # Handle tool results - these become separate messages in OpenAI format
            tool_result = {
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": str(block.get("content", "")),
            }
            tool_results.append(tool_result)

    messages = []

    # Create main message with text content and/or tool calls
    if text_parts or tool_calls:
        main_message = {"role": role}

        if text_parts:
            main_message["content"] = "\n".join(text_parts)

        if tool_calls:
            main_message["tool_calls"] = tool_calls  # type: ignore
            # If there are tool calls but no text content, set empty content
            if not text_parts:
                main_message["content"] = ""

        messages.append(main_message)

    # Add tool result messages
    messages.extend(tool_results)

    # If we only have tool results, return the list
    if len(messages) == 1 and tool_results:
        return messages
    elif len(messages) == 1:
        return messages[0]
    elif len(messages) > 1:
        return messages
    else:
        # Return empty dict instead of None
        return {}


def _transform_tools_to_functions(
    anthropic_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Transform Anthropic tool definitions to OpenAI function format."""
    openai_tools = []

    for tool in anthropic_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }
        openai_tools.append(openai_tool)

    return openai_tools


def _transform_tool_choice(
    anthropic_tool_choice: Union[str, Dict[str, Any]],
) -> Union[str, Dict[str, Any]]:
    """Transform Anthropic tool_choice to OpenAI format."""
    if isinstance(anthropic_tool_choice, str):
        # Handle string values: "auto", "any", "required"
        if anthropic_tool_choice == "auto":
            return "auto"
        elif anthropic_tool_choice == "any" or anthropic_tool_choice == "required":
            return "required"
        else:
            return "auto"

    elif isinstance(anthropic_tool_choice, dict):
        # Handle specific tool choice
        tool_type = anthropic_tool_choice.get("type")
        if tool_type == "tool":
            tool_name = anthropic_tool_choice.get("name")
            if tool_name:
                return {"type": "function", "function": {"name": tool_name}}

    return "auto"


def _extract_text_from_content_blocks(content_blocks: List[Dict[str, Any]]) -> str:
    """Extract text content from content blocks (legacy function)."""
    text_parts = []

    for block in content_blocks:
        if block.get("type") == "text" and "text" in block:
            text_parts.append(block["text"])

    return "\n".join(text_parts) if text_parts else ""
