import logging
from typing import Optional

import httpx

from .messages import Messages

logger = logging.getLogger(__name__)


class AnthropicClient:
    """
    A client that provides Anthropic Messages API compatibility with
    OpenAI-compatible services.

    This client mimics the interface of the official Anthropic SDK while internally
    transforming requests to OpenAI ChatCompletion format and responses back to
    Anthropic format. It provides a drop-in replacement for applications using
    the Anthropic SDK.

    Attributes:
        messages: Messages handler for creating chat completions.

    Example:
        >>> client = AnthropicClient(
        ...     api_key="your-openai-api-key",
        ...     base_url="https://api.openai.com/v1"
        ... )
        >>> response = client.messages.create(
        ...     model="gpt-3.5-turbo",
        ...     max_tokens=100,
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.content[0].text)
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        max_retries: int = 2,
        default_headers: Optional[dict] = None,
        http_client: Optional[httpx.Client] = None,
        **kwargs,
    ):
        """
        Initialize the AnthropicClient.

        Args:
            api_key: OpenAI API key for authentication.
            base_url: Base URL for the OpenAI-compatible service.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of request retries.
            default_headers: Additional headers to include in requests.
            http_client: Custom httpx.Client instance (optional).
            **kwargs: Additional keyword arguments (ignored).
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

        if http_client is None:
            headers = default_headers or {}
            headers.update({"Authorization": f"Bearer {self.api_key}"})
            if max_retries > 0:
                transport = httpx.HTTPTransport(retries=max_retries)
                self.http_client = httpx.Client(
                    timeout=timeout, headers=headers, transport=transport
                )
            else:
                self.http_client = httpx.Client(timeout=timeout, headers=headers)
        else:
            self.http_client = http_client

        self._messages = Messages(
            api_key=self.api_key, base_url=self.base_url, http_client=self.http_client
        )

        logger.info(f"Initialized AnthropicClient with base_url: {self.base_url}")

    @property
    def messages(self) -> Messages:
        return self._messages

    def close(self) -> None:
        if hasattr(self.http_client, "close"):
            self.http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
