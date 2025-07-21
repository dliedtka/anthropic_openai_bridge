# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python library that creates an Anthropic Messages API-compatible interface while internally transforming requests to the OpenAI ChatCompletion API format. The goal is to provide a drop-in replacement for the Anthropic SDK that works with OpenAI-compatible services.

## Architecture

The library follows a transformation pattern:
```
Application Code (Anthropic patterns) → Custom SDK Library → Request Transformation (Anthropic → OpenAI) → HTTP Client → Response Transformation (OpenAI → Anthropic) → Return to Application
```

### Core Components
- **Client Class**: Main entry point mimicking `anthropic.Anthropic`
- **Messages Handler**: Implements `messages.create()` interface
- **Request Transformer**: Converts Anthropic Messages format to OpenAI ChatCompletion
- **Response Transformer**: Converts OpenAI responses back to Anthropic format
- **HTTP Client**: Handles communication with OpenAI-compatible services
- **Error Handler**: Maps OpenAI errors to Anthropic-style exceptions

### Implemented Package Structure
```
anthropic_openai_bridge/
├── __init__.py          # Main exports (clients, exceptions, types, setup_logging)
├── client.py            # AnthropicClient class - main entry point
├── async_client.py      # AsyncAnthropicClient - async version
├── messages.py          # Messages class with create() method
├── async_messages.py    # AsyncMessages - async version
├── streaming.py         # SSE parsing and streaming utilities
├── transformers/
│   ├── __init__.py
│   ├── request.py       # transform_anthropic_to_openai() with tools support
│   └── response.py      # transform_openai_to_anthropic() with tools support
├── exceptions.py        # Anthropic-compatible error classes
├── types.py            # Type definitions, streaming events, tool types
└── utils.py            # Logging utilities and helpers
```

## Development Commands

```bash
# Set up development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"  # Install with dev dependencies from setup.py

# Alternative dependency installation
pip install -r requirements-dev.txt

# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_transformers.py

# Run specific test
python -m pytest tests/test_client.py::TestAnthropicClient::test_successful_message_creation

# Type checking
python -m mypy anthropic_openai_bridge/

# Code formatting
python -m black anthropic_openai_bridge/ tests/
python -m isort anthropic_openai_bridge/ tests/

# Linting
python -m flake8 anthropic_openai_bridge/

# Install in editable mode for development
pip install -e .

# Run tests with coverage
python -m pytest --cov=anthropic_openai_bridge

# Run tests with verbose output
python -m pytest -v
```

## Key Implementation Requirements

### Interface Compatibility
- Must provide the same interface as Anthropic SDK's Messages API
- Support all standard parameters: `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `stop_sequences`, `stream`, `system`, `tools`, `tool_choice`
- Return objects matching Anthropic's response structure exactly

### Critical Transformations

**Phase 1 (Implemented):**
1. **System Messages**: Convert Anthropic's separate `system` parameter to OpenAI's system message format
2. **Message Content**: Handle both string content and content blocks (extract text from blocks)
3. **Parameter Mapping**: Map `temperature`, `top_p`, `max_tokens`, `stop_sequences`
4. **Response Structure**: Transform OpenAI choice format to Anthropic content blocks
5. **Error Mapping**: Map OpenAI HTTP status codes to Anthropic exception types
6. **Stop Reasons**: Convert finish reasons (`stop` → `end_turn`, `length` → `max_tokens`)

**Phase 2 (Implemented):**
- ✅ **Tool Calling**: Complete transformation between Anthropic tool definitions and OpenAI function calling format
- ✅ **Streaming**: Full SSE streaming support with real-time event transformation
- ✅ **Async Support**: AsyncAnthropicClient with full async/await compatibility
- ✅ **Complex Content**: Handle tool_use, tool_result, and mixed content blocks
- ✅ **Message Roles**: Handle role transformations and tool message types

### Dependencies
- **HTTP Client**: `httpx>=0.24.0` (implemented)
- **Type Hints**: Full type annotation support (Python 3.10+)
- **Async Support**: Full async/await compatibility (implemented)

## Testing Strategy

**Implemented Tests:**
- Unit tests for transformation functions (`tests/test_transformers.py`)
- Integration tests with mocked HTTP responses (`tests/test_client.py`) 
- Error handling validation (authentication, rate limiting)
- Parameter validation and edge cases
- ✅ **Streaming functionality tests** (`tests/test_streaming.py`)
- ✅ **Tool calling transformation tests** (`tests/test_tools.py`)
- ✅ **Async functionality tests** (`tests/test_async.py`)

**Test Files:**
- `tests/test_transformers.py`: Tests for request/response transformation logic
- `tests/test_client.py`: Tests for client functionality and error handling
- `tests/test_streaming.py`: Tests for SSE parsing and streaming responses
- `tests/test_tools.py`: Tests for tool calling and function transformation
- `tests/test_async.py`: Tests for async client and streaming functionality

**Testing Coverage:**
- ✅ 30+ comprehensive test cases covering all Phase 2 features
- ✅ Extensive mocking for HTTP responses and streaming data
- ✅ Edge case testing for tool calling and streaming scenarios

## Security Considerations

- Never log API keys or sensitive information
- Validate all input parameters
- Handle malicious or malformed requests gracefully
- Support environment variable configuration for API keys

## Current Implementation Status

**Phase 2 Complete - All Features Implemented:**
- ✅ Basic client and messages handler
- ✅ Core request/response transformation
- ✅ **Streaming support** with Server-Sent Events (SSE)
- ✅ **Tool/function calling** with automatic format conversion
- ✅ **Async support** with AsyncAnthropicClient and AsyncMessages
- ✅ Comprehensive error handling and exception mapping
- ✅ Object-oriented response classes with backward compatibility
- ✅ Complete type definitions and utilities
- ✅ Comprehensive unit and integration tests
- ✅ Package structure with setup.py

**Key Phase 2 Achievements:**
- 🎯 **Complete Anthropic API Compatibility**: All major Anthropic Messages API features now supported
- 🚀 **Production Ready**: Streaming, tools, and async support make it suitable for real applications
- 🔧 **Developer Experience**: Comprehensive type hints, error handling, and backward compatibility
- 🧪 **Well Tested**: 30+ test cases covering all functionality including edge cases

## Project Configuration

- Package dependencies defined in `setup.py` with development extras
- Development dependencies: pytest, black, isort, mypy, flake8
- Python 3.10+ required with httpx>=0.24.0 as core dependency
- .env file contains environment variables for both Anthropic and OpenAI APIs
- Project specifications documented in `project_specification.md`
- API references: `anthropic_messages.md` and `openai_chat_completions.txt`

## Response Object Structure

Response objects follow Anthropic's Message format with both attribute and dict-like access:
- `id`: Message ID (e.g., "msg_123456789")  
- `type`: Always "message"
- `role`: Always "assistant"
- `content`: List of ContentBlock or ToolUse objects
- `model`: Model name used
- `stop_reason`: Reason for stopping ("end_turn", "max_tokens", "tool_use")
- `usage`: Usage object with input_tokens/output_tokens counts

## Error Handling

The library maps OpenAI HTTP errors to Anthropic-compatible exceptions:
- `AuthenticationError`: Invalid API key (401)
- `RateLimitError`: Rate limit exceeded (429) 
- `APIError`: General API errors (400, 500, etc.)
- All exceptions maintain backward compatibility with Anthropic SDK patterns