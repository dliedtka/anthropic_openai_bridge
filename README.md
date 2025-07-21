# Anthropic-OpenAI Bridge

A Python library that provides an Anthropic Messages API-compatible interface while internally transforming requests to the OpenAI ChatCompletion API format. This allows developers to use familiar Anthropic SDK patterns and request/response formats while communicating with OpenAI-compatible services.

## Features

- **Drop-in Compatibility**: Provides the same interface as the Anthropic SDK's Messages API
- **Transparent Translation**: Seamlessly converts between Anthropic Messages and OpenAI ChatCompletion formats
- **Zero Infrastructure**: No proxy servers or external dependencies required
- **Type Safety**: Full type hint support with zero mypy errors for better IDE integration
- **Streaming Support**: Real-time streaming responses with Server-Sent Events (SSE)
- **Tool Calling**: Complete support for function/tool calling with automatic format conversion
- **Async Support**: Full async/await compatibility for high-performance applications
- **High Performance**: ~0.03ms transformation overhead with 29,000+ transformations/second
- **Production Ready**: 91% test coverage with 142 comprehensive test cases
- **Comprehensive Error Handling**: Detailed error mapping and edge case coverage

## Installation

```bash
# Install the package
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from anthropic_openai_bridge import AnthropicClient

# Initialize the client with your OpenAI API key and base URL
client = AnthropicClient(
    api_key="your-openai-api-key",
    base_url="https://api.openai.com/v1"  # or your OpenAI-compatible service URL
)

# Use the familiar Anthropic SDK interface
response = client.messages.create(
    model="gpt-3.5-turbo",  # or any OpenAI-compatible model
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ]
)

print(response.content[0].text)
```

## Usage Examples

### Basic Message Creation

```python
from anthropic_openai_bridge import AnthropicClient

client = AnthropicClient(
    api_key="your-openai-api-key",
    base_url="https://api.openai.com/v1"
)

response = client.messages.create(
    model="gpt-3.5-turbo",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)

print(response.content[0].text)
```

### With System Message

```python
response = client.messages.create(
    model="gpt-3.5-turbo",
    max_tokens=1000,
    system="You are a helpful physics tutor.",
    messages=[
        {"role": "user", "content": "What is quantum entanglement?"}
    ]
)
```

### With Additional Parameters

```python
response = client.messages.create(
    model="gpt-3.5-turbo",
    max_tokens=1000,
    temperature=0.7,
    top_p=0.9,
    stop_sequences=["Human:", "Assistant:"],
    messages=[
        {"role": "user", "content": "Write a short story about a robot."}
    ]
)
```

### Using Context Manager

```python
with AnthropicClient(api_key="your-key", base_url="https://api.openai.com/v1") as client:
    response = client.messages.create(
        model="gpt-3.5-turbo",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.content[0].text)
```

### Streaming Responses

```python
client = AnthropicClient(api_key="your-key", base_url="https://api.openai.com/v1")

stream = client.messages.create(
    model="gpt-3.5-turbo",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for event in stream:
    if hasattr(event, 'delta') and 'text' in event.delta:
        print(event.delta['text'], end='', flush=True)
```

### Tool Calling (Function Calling)

```python
# Define tools
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco, CA'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
]

response = client.messages.create(
    model="gpt-3.5-turbo",
    max_tokens=1000,
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if the assistant wants to call a tool
if response.stop_reason == "tool_use":
    for content in response.content:
        if hasattr(content, 'name'):  # This is a tool use
            print(f"Tool call: {content.name}")
            print(f"Arguments: {content.input}")
```

### Complete Tool Conversation

```python
# Initial request
response = client.messages.create(
    model="gpt-3.5-turbo",
    max_tokens=1000,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

conversation = [
    {"role": "user", "content": "What's the weather in Paris?"},
    {
        "role": "assistant",
        "content": response.content  # Contains text + tool_use
    }
]

# Simulate tool execution (you would call your actual function here)
if response.stop_reason == "tool_use":
    for content in response.content:
        if hasattr(content, 'name'):
            # Add tool result to conversation
            conversation.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": "Sunny, 22°C"  # Your actual tool response
                    }
                ]
            })

# Continue the conversation with tool results
final_response = client.messages.create(
    model="gpt-3.5-turbo",
    max_tokens=1000,
    messages=conversation
)

print(final_response.content[0].text)
```

### Async Support

```python
import asyncio
from anthropic_openai_bridge import AsyncAnthropicClient

async def main():
    client = AsyncAnthropicClient(api_key="your-key", base_url="https://api.openai.com/v1")
    
    response = await client.messages.create(
        model="gpt-3.5-turbo",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    print(response.content[0].text)

asyncio.run(main())
```

### Async Streaming

```python
async def stream_example():
    client = AsyncAnthropicClient(api_key="your-key", base_url="https://api.openai.com/v1")
    
    stream = await client.messages.create(
        model="gpt-3.5-turbo",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Tell me a joke"}],
        stream=True
    )
    
    async for event in stream:
        if hasattr(event, 'delta') and 'text' in event.delta:
            print(event.delta['text'], end='', flush=True)

asyncio.run(stream_example())
```

## API Reference

### AnthropicClient

The main client class that mimics the Anthropic SDK interface.

**Parameters:**
- `api_key` (str): Your OpenAI API key
- `base_url` (str): OpenAI-compatible service URL (default: "https://api.openai.com/v1")
- `timeout` (float): Request timeout in seconds (default: 60.0)
- `max_retries` (int): Maximum number of retries (default: 2)

### AsyncAnthropicClient

The async version of the client for high-performance applications.

**Parameters:**
- `api_key` (str): Your OpenAI API key
- `base_url` (str): OpenAI-compatible service URL (default: "https://api.openai.com/v1")
- `timeout` (float): Request timeout in seconds (default: 60.0)
- `max_retries` (int): Maximum number of retries (default: 2)
- `http_client` (httpx.AsyncClient, optional): Custom async HTTP client

### messages.create()

Create a message using the Anthropic Messages API format.

**Parameters:**
- `model` (str): Model name (passed through to OpenAI)
- `messages` (List[Dict]): List of message objects
- `max_tokens` (int): Maximum tokens in response
- `temperature` (float, optional): Sampling temperature (0-1)
- `top_p` (float, optional): Nucleus sampling parameter
- `stop_sequences` (List[str], optional): Stop sequences
- `system` (str, optional): System message
- `stream` (bool, optional): Enable streaming responses
- `tools` (List[Dict], optional): Tool definitions for function calling
- `tool_choice` (Union[str, Dict], optional): Tool choice strategy ("auto", "required", or specific tool)

## Response Format

### Regular Responses

Responses are Anthropic-compatible Message objects with both attribute and dict-like access:

```python
# Attribute access (recommended)
print(response.id)                    # "msg_123456789"
print(response.type)                  # "message"
print(response.role)                  # "assistant"
print(response.content[0].text)       # "Hello! How can I help you today?"
print(response.model)                 # "gpt-3.5-turbo"
print(response.stop_reason)           # "end_turn"
print(response.usage.input_tokens)    # 10
print(response.usage.output_tokens)   # 25

# Dict-like access (for backward compatibility)
print(response["id"])                 # "msg_123456789"
print(response["content"][0]["text"]) # "Hello! How can I help you today?"
print(response["usage"]["input_tokens"]) # 10
```

### Tool Calling Responses

When using tools, the response content includes both text and tool use blocks:

```python
# Text content
text_content = response.content[0]
print(text_content.type)              # "text"
print(text_content.text)              # "I'll check the weather for you."

# Tool use content
tool_content = response.content[1]
print(tool_content.type)              # "tool_use"
print(tool_content.id)                # "toolu_123456789"
print(tool_content.name)              # "get_weather"
print(tool_content.input)             # {"location": "San Francisco"}
```

### Streaming Responses

Streaming returns an iterator of events:

```python
for event in stream:
    if event.type == "message_start":
        print(f"Started message: {event.message.id}")
    elif event.type == "content_block_start":
        print(f"Started content block: {event.content_block.type}")
    elif event.type == "content_block_delta":
        if "text" in event.delta:
            print(event.delta["text"], end="")
    elif event.type == "message_stop":
        print("\nMessage completed")
```

### Response Object Structure

The response object structure matches the Anthropic Messages API format:
- `id`: Message ID
- `type`: Always "message"
- `role`: Always "assistant"
- `content`: List of ContentBlock or ToolUse objects
- `model`: Model name used
- `stop_reason`: Reason for stopping ("end_turn", "max_tokens", "tool_use", etc.)
- `stop_sequence`: Stop sequence if applicable
- `usage`: Usage object with token counts

## Error Handling

The library maps OpenAI errors to Anthropic-compatible exceptions:

```python
from anthropic_openai_bridge import AnthropicClient, AuthenticationError, RateLimitError

client = AnthropicClient(api_key="invalid-key")

try:
    response = client.messages.create(
        model="gpt-3.5-turbo",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Hello"}]
    )
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
```

### Async Error Handling

```python
from anthropic_openai_bridge import AsyncAnthropicClient, AuthenticationError

client = AsyncAnthropicClient(api_key="invalid-key")

try:
    response = await client.messages.create(
        model="gpt-3.5-turbo",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Hello"}]
    )
except AuthenticationError:
    print("Invalid API key")
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd anthropic_oai_bridge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage (currently 91% coverage)
python -m pytest --cov=anthropic_openai_bridge --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_transformers.py

# Run specific test
python -m pytest tests/test_client.py::TestAnthropicClient::test_successful_message_creation

# Run tests with verbose output
python -m pytest -v

# Run performance benchmarks
python benchmark.py
```

**Current Test Statistics:**
- **142 test cases** across 8 test modules
- **91% code coverage** with detailed missing line reports
- **Edge case testing** for error conditions, malformed data, network issues
- **Performance benchmarks** for transformation overhead measurement

### Code Quality

```bash
# Type checking (currently zero errors)
python -m mypy anthropic_openai_bridge/ --ignore-missing-imports

# Code formatting
python -m black anthropic_openai_bridge/ tests/
python -m isort anthropic_openai_bridge/ tests/

# Linting (currently zero violations)
python -m flake8 anthropic_openai_bridge/ --max-line-length=88
```

## Current Status

### ✅ Phase 3 Complete - Production Ready

**All Features Implemented & Polished:**
- ✅ **Basic Message API**: Full compatibility with Anthropic Messages API
- ✅ **Streaming Support**: Real-time streaming responses with SSE
- ✅ **Tool Calling**: Complete function/tool calling support
- ✅ **Async Support**: Full async/await compatibility
- ✅ **Error Handling**: Comprehensive error mapping with detailed tests
- ✅ **Type Safety**: Zero mypy errors with full type hint coverage
- ✅ **Backward Compatibility**: Dict-like access for legacy code

**Quality & Performance:**
- ✅ **91% Test Coverage**: 142 comprehensive test cases including edge cases
- ✅ **High Performance**: ~0.03ms transformation latency, 29,000+ ops/second
- ✅ **Code Quality**: Zero flake8 violations, comprehensive docstrings
- ✅ **Package Ready**: Complete setup.py with proper metadata for distribution

## Performance

The library has been optimized for minimal overhead:

```
Request Transformation:  ~0.034ms per operation (29,500+ ops/sec)
Response Transformation: ~0.037ms per operation (27,000+ ops/sec)
Memory Overhead:         <10% additional memory usage
JSON Operations:         ~0.026ms serialize, ~0.015ms deserialize
```

To run performance benchmarks yourself:

```bash
python benchmark.py
```

### Minor Limitations

- **Model Mapping**: Model names are passed through as-is (no automatic translation)
- **OpenAI-Specific**: Designed specifically for OpenAI-compatible APIs

## Requirements

- Python 3.10+
- httpx >= 0.24.0

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Support

For issues and questions, please check the project's issue tracker.