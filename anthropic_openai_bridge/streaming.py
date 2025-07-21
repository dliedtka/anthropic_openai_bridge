"""
Streaming utilities for handling Server-Sent Events (SSE) and streaming responses.
"""

import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import httpx

from .types import (
    ContentBlock,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamingEvent,
    StreamingMessage,
    ToolUse,
    Usage,
)

logger = logging.getLogger(__name__)


class SSEParser:
    """Parser for Server-Sent Events (SSE) format."""

    @staticmethod
    def parse_line(line: str) -> Optional[Dict[str, str]]:
        """Parse a single SSE line."""
        line = line.strip()
        if not line or line.startswith(":"):
            return None

        if ":" in line:
            key, value = line.split(":", 1)
            return {key.strip(): value.strip()}
        else:
            return {line: ""}

    @staticmethod
    def parse_event(lines: str) -> Optional[Dict[str, Any]]:
        """Parse SSE event from multiple lines."""
        event = {}
        data_lines = []

        for line in lines.split("\n"):
            parsed = SSEParser.parse_line(line)
            if parsed:
                key, value = next(iter(parsed.items()))
                if key == "data":
                    data_lines.append(value)
                else:
                    event[key] = value

        if data_lines:
            data = "\n".join(data_lines)
            if data == "[DONE]":
                return {"event": "done"}
            try:
                event["data"] = json.loads(data)
            except json.JSONDecodeError:
                event["data"] = data

        return event if event else None


def parse_openai_streaming_response(
    response: httpx.Response,
) -> Iterator[Dict[str, Any]]:
    """Parse OpenAI streaming response and yield events."""
    buffer = ""

    for chunk in response.iter_text():
        buffer += chunk

        while "\n\n" in buffer:
            event_data, buffer = buffer.split("\n\n", 1)
            event = SSEParser.parse_event(event_data)

            if event:
                if event.get("event") == "done":
                    return
                if "data" in event:
                    yield event["data"]


async def parse_openai_streaming_response_async(
    response: httpx.Response,
) -> AsyncIterator[Dict[str, Any]]:
    """Parse OpenAI streaming response asynchronously and yield events."""
    buffer = ""

    async for chunk in response.aiter_text():
        buffer += chunk

        while "\n\n" in buffer:
            event_data, buffer = buffer.split("\n\n", 1)
            event = SSEParser.parse_event(event_data)

            if event:
                if event.get("event") == "done":
                    return
                if "data" in event:
                    yield event["data"]


def transform_openai_stream_to_anthropic(
    openai_events: Iterator[Dict[str, Any]],
) -> Iterator[StreamingEvent]:
    """Transform OpenAI streaming events to Anthropic format."""
    current_message = StreamingMessage()
    content_blocks: List[Union[ContentBlock, ToolUse]] = []

    for event in openai_events:
        try:
            # Handle different types of OpenAI streaming events
            if "choices" not in event:
                continue

            choice = event["choices"][0]
            delta = choice.get("delta", {})

            # Message start
            if delta.get("role"):
                current_message.id = event.get("id", "")
                current_message.model = event.get("model", "")
                current_message.role = delta["role"]
                yield MessageStart(message=current_message)

            # Content delta
            if "content" in delta and delta["content"]:
                content = delta["content"]

                # If this is the first content, start a content block
                if not content_blocks:
                    content_block = ContentBlock(type="text", text="")
                    content_blocks.append(content_block)
                    current_message.content.append(content_block)
                    yield ContentBlockStart(index=0, content_block=content_block)

                # Update the content block (check if it's a ContentBlock, not ToolUse)
                if isinstance(content_blocks[0], ContentBlock):
                    content_blocks[0].text += content
                yield ContentBlockDelta(index=0, delta={"text": content})

            # Tool calls (function calls in OpenAI)
            if "tool_calls" in delta:
                for i, tool_call in enumerate(delta["tool_calls"]):
                    if tool_call.get("function"):
                        func = tool_call["function"]
                        tool_use = ToolUse(
                            id=tool_call.get("id", ""),
                            name=func.get("name", ""),
                            input=(
                                json.loads(func.get("arguments", "{}"))
                                if func.get("arguments")
                                else {}
                            ),
                        )

                        if len(content_blocks) <= i:
                            content_blocks.append(tool_use)
                            current_message.content.append(tool_use)
                            yield ContentBlockStart(index=i, content_block=tool_use)

                        # For tool calls, we typically get the full call at once
                        yield ContentBlockDelta(
                            index=i, delta={"input": tool_use.input}
                        )

            # Finish reason
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                # Stop all active content blocks
                for i in range(len(content_blocks)):
                    yield ContentBlockStop(index=i)

                # Map finish reason
                stop_reason_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                    "function_call": "tool_use",
                    "content_filter": "end_turn",
                }

                current_message.stop_reason = stop_reason_map.get(
                    finish_reason, "end_turn"
                )

                # Usage information
                if "usage" in event:
                    usage_data = event["usage"]
                    current_message.usage = Usage(
                        input_tokens=usage_data.get("prompt_tokens", 0),
                        output_tokens=usage_data.get("completion_tokens", 0),
                    )

                yield MessageDelta(
                    delta={"stop_reason": current_message.stop_reason},
                    usage=current_message.usage,
                )
                yield MessageStop()

        except Exception as e:
            logger.error(f"Error processing streaming event: {e}")
            continue


async def transform_openai_stream_to_anthropic_async(
    openai_events: AsyncIterator[Dict[str, Any]],
) -> AsyncIterator[StreamingEvent]:
    """Transform OpenAI streaming events to Anthropic format asynchronously."""
    current_message = StreamingMessage()
    content_blocks: List[Union[ContentBlock, ToolUse]] = []

    async for event in openai_events:
        try:
            # Handle different types of OpenAI streaming events
            if "choices" not in event:
                continue

            choice = event["choices"][0]
            delta = choice.get("delta", {})

            # Message start
            if delta.get("role"):
                current_message.id = event.get("id", "")
                current_message.model = event.get("model", "")
                current_message.role = delta["role"]
                yield MessageStart(message=current_message)

            # Content delta
            if "content" in delta and delta["content"]:
                content = delta["content"]

                # If this is the first content, start a content block
                if not content_blocks:
                    content_block = ContentBlock(type="text", text="")
                    content_blocks.append(content_block)
                    current_message.content.append(content_block)
                    yield ContentBlockStart(index=0, content_block=content_block)

                # Update the content block (check if it's a ContentBlock, not ToolUse)
                if isinstance(content_blocks[0], ContentBlock):
                    content_blocks[0].text += content
                yield ContentBlockDelta(index=0, delta={"text": content})

            # Tool calls (function calls in OpenAI)
            if "tool_calls" in delta:
                for i, tool_call in enumerate(delta["tool_calls"]):
                    if tool_call.get("function"):
                        func = tool_call["function"]
                        tool_use = ToolUse(
                            id=tool_call.get("id", ""),
                            name=func.get("name", ""),
                            input=(
                                json.loads(func.get("arguments", "{}"))
                                if func.get("arguments")
                                else {}
                            ),
                        )

                        if len(content_blocks) <= i:
                            content_blocks.append(tool_use)
                            current_message.content.append(tool_use)
                            yield ContentBlockStart(index=i, content_block=tool_use)

                        # For tool calls, we typically get the full call at once
                        yield ContentBlockDelta(
                            index=i, delta={"input": tool_use.input}
                        )

            # Finish reason
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                # Stop all active content blocks
                for i in range(len(content_blocks)):
                    yield ContentBlockStop(index=i)

                # Map finish reason
                stop_reason_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                    "function_call": "tool_use",
                    "content_filter": "end_turn",
                }

                current_message.stop_reason = stop_reason_map.get(
                    finish_reason, "end_turn"
                )

                # Usage information
                if "usage" in event:
                    usage_data = event["usage"]
                    current_message.usage = Usage(
                        input_tokens=usage_data.get("prompt_tokens", 0),
                        output_tokens=usage_data.get("completion_tokens", 0),
                    )

                yield MessageDelta(
                    delta={"stop_reason": current_message.stop_reason},
                    usage=current_message.usage,
                )
                yield MessageStop()

        except Exception as e:
            logger.error(f"Error processing streaming event: {e}")
            continue
