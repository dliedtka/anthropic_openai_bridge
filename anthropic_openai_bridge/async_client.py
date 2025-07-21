"""
Async client implementation for the Anthropic-OpenAI bridge.
"""

from typing import Optional

import httpx

from .async_messages import AsyncMessages


class AsyncAnthropicClient:
    """Async version of the AnthropicClient that mimics the Anthropic SDK interface."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        max_retries: int = 2,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._http_client = http_client
        self._messages: Optional[AsyncMessages] = None

    @property
    def messages(self) -> AsyncMessages:
        """Get the async messages handler."""
        if self._messages is None:
            self._messages = AsyncMessages(
                self.api_key, self.base_url, self._http_client
            )
        return self._messages  # mypy: ignore

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._http_client:
            await self._http_client.aclose()
