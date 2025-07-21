"""
Tests for exceptions module.
"""

import pytest
from unittest.mock import Mock

from anthropic_openai_bridge.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
    map_openai_error_to_anthropic,
)


class TestExceptionClasses:
    """Test exception class constructors and attributes."""

    def test_api_error_basic(self):
        """Test APIError basic construction."""
        error = APIError("Test error")
        assert str(error) == "Test error"
        assert error.args == ("Test error",)

    def test_api_error_with_response(self):
        """Test APIError with response."""
        mock_response = Mock()
        error = APIError("Test error", response=mock_response)
        assert str(error) == "Test error"
        assert error.response == mock_response

    def test_authentication_error(self):
        """Test AuthenticationError construction."""
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert isinstance(error, APIError)

    def test_bad_request_error(self):
        """Test BadRequestError construction."""
        error = BadRequestError("Bad request")
        assert str(error) == "Bad request"
        assert isinstance(error, APIError)

    def test_not_found_error(self):
        """Test NotFoundError construction."""
        error = NotFoundError("Resource not found")
        assert str(error) == "Resource not found"
        assert isinstance(error, APIError)

    def test_permission_denied_error(self):
        """Test PermissionDeniedError construction."""
        error = PermissionDeniedError("Permission denied")
        assert str(error) == "Permission denied"
        assert isinstance(error, APIError)

    def test_conflict_error(self):
        """Test ConflictError construction."""
        error = ConflictError("Conflict detected")
        assert str(error) == "Conflict detected"
        assert isinstance(error, APIError)

    def test_unprocessable_entity_error(self):
        """Test UnprocessableEntityError construction."""
        error = UnprocessableEntityError("Validation error")
        assert str(error) == "Validation error"
        assert isinstance(error, APIError)

    def test_rate_limit_error(self):
        """Test RateLimitError construction."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, APIError)

    def test_internal_server_error(self):
        """Test InternalServerError construction."""
        error = InternalServerError("Server error")
        assert str(error) == "Server error"
        assert isinstance(error, APIError)


class TestMapOpenAIErrorToAnthropic:
    """Test the error mapping function."""

    def test_map_401_to_authentication_error(self):
        """Test mapping 401 status code to AuthenticationError."""
        error_data = {"error": {"message": "Incorrect API key"}}
        
        error = map_openai_error_to_anthropic(401, error_data)
        
        assert isinstance(error, AuthenticationError)
        assert "Incorrect API key" in str(error)

    def test_map_400_to_bad_request_error(self):
        """Test mapping 400 status code to BadRequestError."""
        error_data = {"error": {"message": "Invalid request"}}
        
        error = map_openai_error_to_anthropic(400, error_data)
        
        assert isinstance(error, BadRequestError)
        assert "Invalid request" in str(error)

    def test_map_403_to_permission_denied_error(self):
        """Test mapping 403 status code to PermissionDeniedError."""
        error_data = {"error": {"message": "Forbidden"}}
        
        error = map_openai_error_to_anthropic(403, error_data)
        
        assert isinstance(error, PermissionDeniedError)
        assert "Forbidden" in str(error)

    def test_map_404_to_not_found_error(self):
        """Test mapping 404 status code to NotFoundError."""
        error_data = {"error": {"message": "Not found"}}
        
        error = map_openai_error_to_anthropic(404, error_data)
        
        assert isinstance(error, NotFoundError)
        assert "Not found" in str(error)

    def test_map_409_to_conflict_error(self):
        """Test mapping 409 status code to ConflictError."""
        error_data = {"error": {"message": "Conflict"}}
        
        error = map_openai_error_to_anthropic(409, error_data)
        
        assert isinstance(error, ConflictError)
        assert "Conflict" in str(error)

    def test_map_422_to_unprocessable_entity_error(self):
        """Test mapping 422 status code to UnprocessableEntityError."""
        error_data = {"error": {"message": "Validation failed"}}
        
        error = map_openai_error_to_anthropic(422, error_data)
        
        assert isinstance(error, UnprocessableEntityError)
        assert "Validation failed" in str(error)

    def test_map_429_to_rate_limit_error(self):
        """Test mapping 429 status code to RateLimitError."""
        error_data = {"error": {"message": "Rate limit exceeded"}}
        
        error = map_openai_error_to_anthropic(429, error_data)
        
        assert isinstance(error, RateLimitError)
        assert "Rate limit exceeded" in str(error)

    def test_map_500_to_internal_server_error(self):
        """Test mapping 500 status code to InternalServerError."""
        error_data = {"error": {"message": "Internal server error"}}
        
        error = map_openai_error_to_anthropic(500, error_data)
        
        assert isinstance(error, InternalServerError)
        assert "Internal server error" in str(error)

    def test_map_unknown_status_to_api_error(self):
        """Test mapping unknown status code to generic APIError."""
        error_data = {"error": {"message": "I'm a teapot"}}
        
        error = map_openai_error_to_anthropic(418, error_data)
        
        assert isinstance(error, APIError)
        assert not isinstance(error, (AuthenticationError, BadRequestError))
        assert "I'm a teapot" in str(error)

    def test_map_error_without_error_data(self):
        """Test mapping error when no error data is provided."""
        error = map_openai_error_to_anthropic(401)
        
        assert isinstance(error, AuthenticationError)
        assert "Unknown error" in str(error)

    def test_map_error_with_missing_error_message(self):
        """Test mapping error when JSON doesn't have expected structure."""
        error_data = {"unexpected": "structure"}
        
        error = map_openai_error_to_anthropic(400, error_data)
        
        assert isinstance(error, BadRequestError)
        assert "Unknown error" in str(error)

    def test_map_error_with_response_object(self):
        """Test that mapped errors can include response object."""
        mock_response = Mock()
        error_data = {"error": {"message": "Too many requests"}}
        
        error = map_openai_error_to_anthropic(429, error_data, mock_response)
        
        assert isinstance(error, RateLimitError)
        assert error.response == mock_response

    def test_map_error_with_detailed_error_info(self):
        """Test mapping error with detailed error information."""
        error_data = {
            "error": {
                "message": "Invalid parameter 'temperature'",
                "type": "invalid_request_error",
                "param": "temperature",
                "code": "invalid_parameter"
            }
        }
        
        error = map_openai_error_to_anthropic(400, error_data)
        
        assert isinstance(error, BadRequestError)
        assert "Invalid parameter 'temperature'" in str(error)