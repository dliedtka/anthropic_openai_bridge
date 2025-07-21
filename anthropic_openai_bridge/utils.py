import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def log_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Any] = None,
) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Request: {method} {url}")
        if headers:
            sanitized_headers = {
                k: ("***" if "authorization" in k.lower() else v)
                for k, v in headers.items()
            }
            logger.debug(f"Headers: {sanitized_headers}")
        if data:
            if isinstance(data, dict):
                data_str = json.dumps(data, indent=2)
            else:
                data_str = str(data)
            logger.debug(f"Data: {data_str}")


def log_response(
    status_code: int, data: Optional[Any] = None, duration: Optional[float] = None
) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Response: {status_code}")
        if duration:
            logger.debug(f"Duration: {duration:.3f}s")
        if data:
            if isinstance(data, dict):
                data_str = json.dumps(data, indent=2)
            else:
                data_str = str(data)
            logger.debug(f"Response data: {data_str}")


def measure_time():
    return time.time()


def sanitize_for_logging(data: Any) -> Any:
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if any(
                sensitive in key.lower()
                for sensitive in ["api_key", "token", "authorization", "password"]
            ):
                sanitized[key] = "***"
            elif isinstance(value, (dict, list)):
                sanitized[key] = sanitize_for_logging(value)
            else:
                sanitized[key] = value
        return sanitized
    elif isinstance(data, list):
        return [sanitize_for_logging(item) for item in data]
    else:
        return data
