# Anthropic-to-OpenAI SDK Bridge - Project Specification

## Overview

Create a Python library that provides an Anthropic Messages API-compatible interface while internally transforming requests to the OpenAI ChatCompletion API format. This allows developers to use Anthropic SDK patterns and request/response formats while communicating with an OpenAI-compatible service.

## Goals

- **Drop-in Compatibility**: Provide the same interface as the Anthropic SDK's Messages API
- **Transparent Translation**: Seamlessly convert between Anthropic Messages and OpenAI ChatCompletion formats
- **Zero Infrastructure**: No proxy servers or external dependencies required
- **Developer Experience**: Maintain familiar Anthropic SDK patterns and error handling

## Architecture

### High-Level Design
```
Application Code (using Anthropic patterns)
           ↓
Custom SDK Library (mimics Anthropic interface)
           ↓
Request Transformation (Anthropic → OpenAI)
           ↓
HTTP Client (to OpenAI-compatible service)
           ↓
Response Transformation (OpenAI → Anthropic)
           ↓
Return to Application (Anthropic format)
```

### Core Components

1. **Client Class**: Main entry point that mimics `anthropic.Anthropic`
2. **Messages Handler**: Implements the `messages.create()` interface
3. **Request Transformer**: Converts Anthropic Messages format to OpenAI ChatCompletion
4. **Response Transformer**: Converts OpenAI ChatCompletion response to Anthropic Messages
5. **HTTP Client**: Handles communication with the OpenAI-compatible service
6. **Error Handler**: Maps OpenAI errors to Anthropic-style exceptions

## Interface Specification

### Primary Interface
The library should provide this interface:

```python
# Target usage pattern (same as Anthropic SDK)
client = CustomAnthropicClient(
    api_key="your-api-key",
    base_url="https://your-openai-service.com/v1"
)

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ]
)
```

### Required Methods and Properties

#### Client Class
- `__init__(api_key, base_url, **kwargs)`
- `messages` property that returns a Messages handler instance

#### Messages Handler
- `create(**kwargs)` - Main method for creating messages
- Support for all standard Anthropic Messages API parameters:
  - `model` (string)
  - `messages` (array)
  - `max_tokens` (integer)
  - `temperature` (float, optional)
  - `top_p` (float, optional)
  - `stop_sequences` (array, optional)
  - `stream` (boolean, optional)
  - `system` (string, optional)
  - `tools` (array, optional)
  - `tool_choice` (object, optional)

#### Response Objects
- Return objects that match Anthropic's response structure
- Support for both regular and streaming responses
- Proper typing hints

## Transformation Requirements

### Request Transformation (Anthropic → OpenAI)

#### Core Mappings
- **Model**: Map Anthropic model names to OpenAI-compatible equivalents
- **Messages**: Convert message format and role handling
- **System Messages**: Transform Anthropic's separate `system` parameter to OpenAI's system message
- **Parameters**: Map temperature, top_p, max_tokens, stop sequences
- **Tools**: Convert Anthropic tool definitions to OpenAI function calling format
- **Streaming**: Handle streaming parameter differences

#### Special Considerations
- Handle Anthropic's role restrictions (alternating user/assistant)
- Convert tool use and tool result message types
- Map unsupported parameters gracefully

### Response Transformation (OpenAI → Anthropic)

#### Core Mappings
- **Message Content**: Convert OpenAI choice format to Anthropic content blocks
- **Tool Calls**: Transform OpenAI function calls to Anthropic tool use format
- **Usage Stats**: Map token usage information
- **Stop Reasons**: Convert OpenAI finish reasons to Anthropic stop reasons
- **Streaming**: Handle streaming response format differences

#### Response Structure
Ensure responses match Anthropic's exact schema:
- `id`, `type`, `role`, `content`, `model`, `stop_reason`, `stop_sequence`, `usage`

## Error Handling

### Error Mapping
- Map OpenAI HTTP status codes to appropriate Anthropic exception types
- Preserve error messages and details where possible
- Handle rate limiting, authentication, and validation errors

### Exception Types
Implement Anthropic-compatible exceptions:
- `AuthenticationError`
- `RateLimitError`
- `BadRequestError`
- `InternalServerError`
- `APIError` (base class)

## Technical Requirements

### Dependencies
- **HTTP Client**: `httpx` or `requests` for API communication
- **JSON Handling**: Built-in `json` module
- **Type Hints**: Full type annotation support
- **Async Support**: Optional async/await compatibility

### Configuration
- Support for custom base URLs
- API key configuration
- Timeout and retry settings
- Custom headers support

### Logging
- Structured logging for debugging transformations
- Optional request/response logging
- Performance metrics

## Streaming Support

### Implementation Requirements
- Support for Server-Sent Events (SSE)
- Proper event parsing and transformation
- Handle partial message assembly
- Error handling during streaming

### Interface
- Return appropriate generator/async generator objects
- Match Anthropic's streaming response format
- Support streaming interruption and cleanup

## Testing Strategy

### Unit Tests
- Test each transformation function independently
- Mock HTTP responses for predictable testing
- Edge cases and error conditions
- Type checking and validation

### Integration Tests
- End-to-end testing with actual OpenAI-compatible service
- Streaming functionality verification
- Error handling validation
- Performance benchmarking

### Test Coverage
- Aim for >90% code coverage
- Test all supported API parameters
- Verify response format compliance

## Documentation Requirements

### API Documentation
- Complete method and parameter documentation
- Usage examples for common scenarios
- Migration guide from official Anthropic SDK
- Error handling examples

### README
- Installation instructions
- Quick start guide
- Configuration options
- Troubleshooting section

## Performance Considerations

### Optimization Targets
- Minimal transformation overhead
- Efficient JSON parsing and serialization
- Connection pooling for HTTP requests
- Memory-efficient streaming

### Monitoring
- Request/response latency tracking
- Transformation time measurements
- Error rate monitoring

## Security Considerations

### API Key Handling
- Secure storage and transmission of API keys
- No logging of sensitive information
- Support for environment variable configuration

### Input Validation
- Validate all input parameters
- Sanitize user-provided content
- Handle malicious or malformed requests gracefully

## Deliverables

### Code Deliverables
1. **Core Library**: Complete implementation with all required interfaces
2. **Type Stubs**: Full type hint support
3. **Tests**: Comprehensive test suite
4. **Examples**: Sample applications demonstrating usage

### Documentation Deliverables
1. **API Reference**: Complete method documentation
2. **User Guide**: Installation and usage instructions
3. **Developer Guide**: Architecture and extension points
4. **Migration Guide**: Moving from official Anthropic SDK

### Package Structure
```
anthropic_openai_bridge/
├── __init__.py
├── client.py
├── messages.py
├── transformers/
│   ├── __init__.py
│   ├── request.py
│   └── response.py
├── exceptions.py
├── types.py
└── utils.py
```

## Success Criteria

1. **Functional**: All core Anthropic Messages API features work correctly
2. **Compatible**: Drop-in replacement for basic use cases
3. **Reliable**: Robust error handling and edge case management
4. **Performant**: Minimal overhead compared to direct API calls
5. **Maintainable**: Clean, well-documented, and testable code
6. **Typed**: Full type hint support for IDE integration

## Timeline Considerations

### Phase 1: Core Implementation (Week 1-2)
- Basic client and messages handler
- Core request/response transformation
- Simple error handling

### Phase 2: Advanced Features (Week 3)
- Streaming support
- Tool/function calling
- Comprehensive error mapping

### Phase 3: Polish & Testing (Week 4)
- Complete test coverage
- Documentation
- Performance optimization
- Package preparation

## Questions for Engineering Team

1. **Async Support**: Do we need async/await compatibility?
    - This would be great.
2. **Model Mapping**: How should we handle model name translation?
    - Model name should stay the same. 
3. **Version Compatibility**: Which Python versions should we support?
    - Python 3.10+ is fine.
4. **Package Management**: PyPI publication requirements?
    - No specific requirements. Not sure we will even publish, as this is for personal use, but it would be cool to do so if it works well.
5. **CI/CD**: Testing and deployment pipeline needs?
    - This is for personal use. Tests would be great, but CI/CD is not necessary.