from typing import Any, Dict, Optional

import httpx


class APIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
    ) -> None:
        super().__init__(message)
        self.request = request
        self.response = response


class AuthenticationError(APIError):
    pass


class BadRequestError(APIError):
    pass


class ConflictError(APIError):
    pass


class InternalServerError(APIError):
    pass


class NotFoundError(APIError):
    pass


class PermissionDeniedError(APIError):
    pass


class RateLimitError(APIError):
    pass


class UnprocessableEntityError(APIError):
    pass


def map_openai_error_to_anthropic(
    status_code: int,
    error_data: Optional[Dict[str, Any]] = None,
    response: Optional[httpx.Response] = None,
) -> APIError:
    message = "Unknown error"

    if error_data:
        if "error" in error_data:
            if isinstance(error_data["error"], dict):
                message = error_data["error"].get("message", message)
            else:
                message = str(error_data["error"])
        elif "message" in error_data:
            message = error_data["message"]

    error_class = APIError

    if status_code == 400:
        error_class = BadRequestError
    elif status_code == 401:
        error_class = AuthenticationError
    elif status_code == 403:
        error_class = PermissionDeniedError
    elif status_code == 404:
        error_class = NotFoundError
    elif status_code == 409:
        error_class = ConflictError
    elif status_code == 422:
        error_class = UnprocessableEntityError
    elif status_code == 429:
        error_class = RateLimitError
    elif status_code >= 500:
        error_class = InternalServerError

    return error_class(message, response=response)
