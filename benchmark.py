#!/usr/bin/env python3
"""
Performance benchmarks for the Anthropic-OpenAI bridge.
"""

import json
import time
from typing import Dict, Any

from anthropic_openai_bridge.transformers import (
    transform_anthropic_to_openai,
    transform_openai_to_anthropic,
)


def benchmark_request_transformation():
    """Benchmark request transformation performance."""
    # Sample Anthropic request
    anthropic_request = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you today?"},
            {"role": "user", "content": "Can you help me write a Python function?"}
        ],
        "system": "You are a helpful programming assistant.",
        "temperature": 0.7,
        "top_p": 0.9,
        "tools": [
            {
                "name": "calculate",
                "description": "Perform basic arithmetic calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
                        "precision": {"type": "integer", "description": "Number of decimal places", "default": 2}
                    },
                    "required": ["expression"]
                }
            }
        ]
    }
    
    # Warm up
    for _ in range(10):
        transform_anthropic_to_openai(anthropic_request)
    
    # Benchmark
    iterations = 1000
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        result = transform_anthropic_to_openai(anthropic_request)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"Request Transformation Benchmark:")
    print(f"  {iterations} iterations in {duration:.4f} seconds")
    print(f"  Average: {duration/iterations*1000:.4f} ms per transformation")
    print(f"  Rate: {iterations/duration:.0f} transformations/second")


def benchmark_response_transformation():
    """Benchmark response transformation performance."""
    # Sample OpenAI response
    openai_response = {
        "id": "chatcmpl-123456789",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here's a Python function that demonstrates basic functionality:\n\n```python\ndef greet(name):\n    return f'Hello, {name}!'\n```\n\nThis function takes a name as input and returns a greeting message.",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": "{\"expression\": \"2+2\", \"precision\": 0}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 45,
            "completion_tokens": 89,
            "total_tokens": 134
        }
    }
    
    # Warm up
    for _ in range(10):
        transform_openai_to_anthropic(openai_response)
    
    # Benchmark
    iterations = 1000
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        result = transform_openai_to_anthropic(openai_response)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"\nResponse Transformation Benchmark:")
    print(f"  {iterations} iterations in {duration:.4f} seconds")
    print(f"  Average: {duration/iterations*1000:.4f} ms per transformation")
    print(f"  Rate: {iterations/duration:.0f} transformations/second")


def benchmark_json_operations():
    """Benchmark JSON serialization/deserialization overhead."""
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello world"} for _ in range(10)],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    # Benchmark serialization
    iterations = 10000
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        json_str = json.dumps(data)
    
    end_time = time.perf_counter()
    serialize_duration = end_time - start_time
    
    # Benchmark deserialization
    json_str = json.dumps(data)
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        parsed = json.loads(json_str)
    
    end_time = time.perf_counter()
    deserialize_duration = end_time - start_time
    
    print(f"\nJSON Operations Benchmark:")
    print(f"  Serialization: {serialize_duration/iterations*1000:.4f} ms per operation")
    print(f"  Deserialization: {deserialize_duration/iterations*1000:.4f} ms per operation")


def benchmark_memory_usage():
    """Basic memory usage estimation."""
    import sys
    
    # Sample objects
    anthropic_request = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "Hello world"}],
        "system": "You are helpful"
    }
    
    openai_request = transform_anthropic_to_openai(anthropic_request)
    
    print(f"\nMemory Usage Estimation:")
    print(f"  Original Anthropic request: ~{sys.getsizeof(json.dumps(anthropic_request))} bytes")
    print(f"  Transformed OpenAI request: ~{sys.getsizeof(json.dumps(openai_request))} bytes")
    
    # Sample response
    openai_response = {
        "choices": [{"message": {"content": "Hello there!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }
    
    anthropic_response = transform_openai_to_anthropic(openai_response)
    
    print(f"  OpenAI response: ~{sys.getsizeof(json.dumps(openai_response))} bytes")
    print(f"  Transformed Anthropic response: ~{len(repr(anthropic_response))} bytes")


if __name__ == "__main__":
    print("Anthropic-OpenAI Bridge Performance Benchmarks")
    print("=" * 50)
    
    benchmark_request_transformation()
    benchmark_response_transformation()
    benchmark_json_operations()
    benchmark_memory_usage()
    
    print(f"\nBenchmark completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")