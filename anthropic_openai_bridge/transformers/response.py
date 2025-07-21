import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from ..types import (
    OPENAI_TO_ANTHROPIC_STOP_REASON_MAP,
    ContentBlock,
    Message,
    ToolUse,
    Usage,
)

logger = logging.getLogger(__name__)


def transform_openai_to_anthropic(openai_response: Dict[str, Any]) -> Message:
    """
    Transform OpenAI ChatCompletion response to Anthropic Messages format.

    This function converts response data from OpenAI's ChatCompletion API format
    to Anthropic's Messages API format, handling text content, tool calls, usage
    statistics, and finish reasons.

    Args:
        openai_response: Dictionary containing OpenAI ChatCompletion response.
                        Expected to have 'choices' list with message data and
                        'usage' object with token counts.

    Returns:
        Message object compatible with Anthropic Messages API format,
        including id, content blocks, model, stop_reason, and usage.

    Example:
        >>> openai_resp = {
        ...     "choices": [{"message": {"content": "Hello!"},
        ...                  "finish_reason": "stop"}],
        ...     "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        ... }
        >>> msg = transform_openai_to_anthropic(openai_resp)
        >>> msg.content[0].text
        'Hello!'
    """
    choice = openai_response["choices"][0] if openai_response.get("choices") else {}
    message = choice.get("message", {})

    content_blocks: List[Union[ContentBlock, ToolUse]] = []

    # Handle text content
    if message.get("content"):
        content_blocks.append(ContentBlock(type="text", text=message["content"]))

    # Handle tool calls (function calls in OpenAI)
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            if tool_call.get("function"):
                func = tool_call["function"]
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}

                tool_use = ToolUse(
                    id=tool_call.get("id", ""),
                    name=func.get("name", ""),
                    input=arguments,
                )
                content_blocks.append(tool_use)

    finish_reason = choice.get("finish_reason")
    stop_reason = _map_finish_reason_to_stop_reason(finish_reason)

    usage_data = openai_response.get("usage", {})
    usage = Usage(
        input_tokens=usage_data.get("prompt_tokens", 0),
        output_tokens=usage_data.get("completion_tokens", 0),
    )

    anthropic_response = Message(
        id=openai_response.get("id", f"msg_{uuid.uuid4().hex}"),
        content=content_blocks,
        model=openai_response.get("model", "unknown"),
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=usage,
    )

    logger.debug(f"Transformed OpenAI response to Anthropic: {anthropic_response}")
    return anthropic_response


def _map_finish_reason_to_stop_reason(finish_reason: Optional[str]) -> Optional[str]:
    if finish_reason is None:
        return None

    return OPENAI_TO_ANTHROPIC_STOP_REASON_MAP.get(finish_reason, "end_turn")
