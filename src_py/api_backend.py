from typing import Any
import os
from codebase_rs import ApiModel

def create_api_backend(
    api_model: ApiModel,
) -> Any:
    try:
        from openai import AsyncOpenAI
        # import httpx
    except ImportError:
        raise ImportError(
            "API backend requires the openai library. "
            "Install with: pip install openai"
        )

    # Configure httpx with connection pooling
    # Note: OpenAI's API has server-side rate limits (RPM/TPM) that are more restrictive
    # than connection limits. The client will automatically retry on 429 errors.
    # http_client = httpx.AsyncClient(
    #     limits=httpx.Limits(
    #         max_connections=100,      # Reduced to 100 to avoid overwhelming the API
    #         max_keepalive_connections=50,  # Keep half for connection reuse
    #     ),
    #     timeout=httpx.Timeout(30.0, read=90.0)  # Reasonable timeouts: 30s connect, 90s read
    # )
    api_key_name = api_model.api_key_name()
    base_url = api_model.base_url()
    api_key = os.getenv(api_key_name)
    if not api_key:
        raise ValueError(f"API key for model {api_model} not found. Please set the environment variable '{api_key_name}'.")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        # http_client=http_client,
        # max_retries=3,  # Retry on rate limit errors, but not too many (was 5)
        # timeout=90.0    # Overall timeout for requests (was 300s)
    )
    print(f"Created API backend for model {api_model.to_string()} with auto-retry on rate limits")
    return client



# The problem is that API models have different calling conventions

# But HuggingFace and Vllm models share the same conventions

# We may simply discard the general generate function for API backends


# Polymorphism can be implemented solely on Python side

# Rust side provides the backend object

# Rust only needs the outer interface

# Interface methods can be dispatched to Python implementations