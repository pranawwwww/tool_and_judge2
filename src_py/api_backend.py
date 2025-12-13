




def create_api_backend(
    model_name: str,
    api_key: str,
) -> any:
    try:
        from openai import AsyncOpenAI
        import httpx
    except ImportError:
        raise ImportError(
            "API backend requires the openai library. "
            "Install with: pip install openai"
        )

    # Configure httpx with connection pooling
    # Note: OpenAI's API has server-side rate limits (RPM/TPM) that are more restrictive
    # than connection limits. The client will automatically retry on 429 errors.
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=100,      # Reduced to 100 to avoid overwhelming the API
            max_keepalive_connections=50,  # Keep half for connection reuse
        ),
        timeout=httpx.Timeout(60.0, read=300.0)  # Longer read timeout for rate-limited responses
    )

    client = AsyncOpenAI(
        api_key=api_key,
        http_client=http_client,
        max_retries=5,  # Automatically retry on rate limit errors (429)
        timeout=300.0   # Overall timeout for requests
    )
    print(f"Created API backend for model {model_name} with auto-retry on rate limits")
    return client



# The problem is that API models have different calling conventions

# But HuggingFace and Vllm models share the same conventions

# We may simply discard the general generate function for API backends


# Polymorphism can be implemented solely on Python side

# Rust side provides the backend object

# Rust only needs the outer interface

# Interface methods can be dispatched to Python implementations