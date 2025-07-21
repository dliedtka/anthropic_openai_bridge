"""
Tests for utils module.
"""

import json
import logging
import time
from unittest.mock import Mock, patch

import pytest

from anthropic_openai_bridge.utils import log_request, log_response, measure_time, setup_logging


class TestLogging:
    """Test logging utility functions."""

    def test_setup_logging_default_level(self):
        """Test setup_logging with default level."""
        with patch("anthropic_openai_bridge.utils.logging.basicConfig") as mock_config:
            setup_logging()
            mock_config.assert_called_once()
            call_kwargs = mock_config.call_args.kwargs
            assert call_kwargs["level"] == logging.INFO
            assert call_kwargs["format"] == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            assert len(call_kwargs["handlers"]) == 1
            assert isinstance(call_kwargs["handlers"][0], logging.StreamHandler)

    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom level."""
        with patch("anthropic_openai_bridge.utils.logging.basicConfig") as mock_config:
            setup_logging(logging.DEBUG)
            mock_config.assert_called_once()
            call_kwargs = mock_config.call_args.kwargs
            assert call_kwargs["level"] == logging.DEBUG
            assert call_kwargs["format"] == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            assert len(call_kwargs["handlers"]) == 1
            assert isinstance(call_kwargs["handlers"][0], logging.StreamHandler)


class TestLogRequest:
    """Test log_request function."""

    def test_log_request_debug_disabled(self):
        """Test log_request when debug logging is disabled."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = False
            
            log_request("GET", "https://example.com", {"key": "value"}, {"content": "test"})
            
            mock_logger.debug.assert_not_called()

    def test_log_request_with_method_and_url_only(self):
        """Test log_request with only method and URL."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            
            log_request("GET", "https://example.com")
            
            mock_logger.debug.assert_any_call("Request: GET https://example.com")

    def test_log_request_with_headers(self):
        """Test log_request with headers."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            headers = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
            
            log_request("POST", "https://example.com", headers)
            
            expected_headers = {
                "Authorization": "***",  # Authorization header is masked
                "Content-Type": "application/json"
            }
            mock_logger.debug.assert_any_call(f"Headers: {expected_headers}")

    def test_log_request_with_dict_data(self):
        """Test log_request with dictionary data."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            data = {"key": "value", "nested": {"item": "test"}}
            
            log_request("POST", "https://example.com", None, data)
            
            mock_logger.debug.assert_any_call(f"Data: {json.dumps(data, indent=2)}")

    def test_log_request_with_string_data(self):
        """Test log_request with string data."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            data = "string data"
            
            log_request("POST", "https://example.com", None, data)
            
            mock_logger.debug.assert_any_call("Data: string data")

    def test_log_request_with_authorization_header_masking(self):
        """Test that Authorization headers are masked."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            headers = {"authorization": "Bearer token123", "x-api-key": "secret456"}
            
            log_request("GET", "https://example.com", headers)
            
            # Check that authorization header is masked but others aren't
            expected_headers = {
                "authorization": "***",  # Only authorization is masked by default
                "x-api-key": "secret456"  # Other headers are not masked by the current implementation
            }
            mock_logger.debug.assert_any_call(f"Headers: {expected_headers}")
            
            # Verify sensitive data is not in the call arguments
            call_args = mock_logger.debug.call_args_list
            headers_call = [call for call in call_args if "Headers:" in str(call)][0]
            headers_str = str(headers_call)
            assert "token123" not in headers_str  # The actual token should be masked


class TestLogResponse:
    """Test log_response function."""

    def test_log_response_debug_disabled(self):
        """Test log_response when debug logging is disabled."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = False
            
            log_response(200, {"result": "success"}, 1.5)
            
            mock_logger.debug.assert_not_called()

    def test_log_response_status_only(self):
        """Test log_response with status code only."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            
            log_response(200)
            
            mock_logger.debug.assert_any_call("Response: 200")

    def test_log_response_with_duration(self):
        """Test log_response with duration."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            
            log_response(200, duration=2.456)
            
            mock_logger.debug.assert_any_call("Duration: 2.456s")

    def test_log_response_with_dict_data(self):
        """Test log_response with dictionary data."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            data = {"id": "msg_123", "content": "Hello"}
            
            log_response(200, data)
            
            mock_logger.debug.assert_any_call(f"Response data: {json.dumps(data, indent=2)}")

    def test_log_response_with_string_data(self):
        """Test log_response with string data."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            data = "response string"
            
            log_response(200, data)
            
            mock_logger.debug.assert_any_call("Response data: response string")

    def test_log_response_complete(self):
        """Test log_response with all parameters."""
        with patch("anthropic_openai_bridge.utils.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            data = {"status": "ok"}
            
            log_response(200, data, 1.234)
            
            mock_logger.debug.assert_any_call("Response: 200")
            mock_logger.debug.assert_any_call("Duration: 1.234s")
            mock_logger.debug.assert_any_call(f"Response data: {json.dumps(data, indent=2)}")


class TestMeasureTime:
    """Test measure_time function."""

    def test_measure_time_returns_float(self):
        """Test that measure_time returns a float timestamp."""
        result = measure_time()
        assert isinstance(result, float)
        assert result > 0

    def test_measure_time_advances(self):
        """Test that measure_time returns increasing values."""
        time1 = measure_time()
        time.sleep(0.01)  # Small delay
        time2 = measure_time()
        assert time2 > time1

    @patch("anthropic_openai_bridge.utils.time")
    def test_measure_time_calls_time_module(self, mock_time):
        """Test that measure_time calls time.time()."""
        mock_time.time.return_value = 1234567890.123
        
        result = measure_time()
        
        mock_time.time.assert_called_once()
        assert result == 1234567890.123