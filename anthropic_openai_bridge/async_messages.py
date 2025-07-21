"""
Async messages handler for the Anthropic-OpenAI bridge.
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from .exceptions import APIError, map_openai_error_to_anthropic
from .streaming import (
    parse_openai_streaming_response_async,
    transform_openai_stream_to_anthropic_async,
)
from .transformers import transform_anthropic_to_openai, transform_openai_to_anthropic
from .types import Message, StreamingEvent
from .utils import log_request, log_response, measure_time

logger = logging.getLogger(__name__)


class AsyncMessages:
    """Async messages handler that implements the messages.create() interface."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client or httpx.AsyncClient()

    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = None,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Message, AsyncIterator[StreamingEvent]]:
        """Create a message using the Anthropic Messages API format asynchronously."""
        anthropic_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if temperature is not None:
            anthropic_params["temperature"] = temperature
        if top_p is not None:
            anthropic_params["top_p"] = top_p
        if stop_sequences is not None:
            anthropic_params["stop_sequences"] = stop_sequences
        if stream is not None:
            anthropic_params["stream"] = stream
        if system is not None:
            anthropic_params["system"] = system
        if tools is not None:
            anthropic_params["tools"] = tools
        if tool_choice is not None:
            anthropic_params["tool_choice"] = tool_choice

        anthropic_params.update(kwargs)

        # Streaming and tools are now supported in Phase 2

        openai_params = transform_anthropic_to_openai(anthropic_params)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        start_time = measure_time()
        log_request("POST", url, headers, openai_params)

        try:
            response = await self.http_client.post(
                url, headers=headers, json=openai_params, timeout=60.0
            )

            duration = measure_time() - start_time

            if response.status_code != 200:
                error_data = None
                try:
                    error_data = response.json()
                except Exception:
                    pass

                log_response(response.status_code, error_data, duration)
                raise map_openai_error_to_anthropic(
                    response.status_code, error_data, response
                )

            # Handle streaming vs non-streaming responses
            if stream:
                log_response(response.status_code, {"streaming": True}, duration)
                # Parse OpenAI streaming response and transform to Anthropic format
                openai_events = parse_openai_streaming_response_async(response)
                return transform_openai_stream_to_anthropic_async(openai_events)
            else:
                openai_response = response.json()
                log_response(response.status_code, openai_response, duration)

                anthropic_response = transform_openai_to_anthropic(openai_response)
                return anthropic_response

        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            raise map_openai_error_to_anthropic(500, {"error": {"message": str(e)}})
        except APIError:
            # Re-raise API errors (AuthenticationError, RateLimitError, etc.)
            raise
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            raise map_openai_error_to_anthropic(500, {"error": {"message": str(e)}})
