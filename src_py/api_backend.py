




def create_api_backend(
    model_name: str,
    api_key: str,
) -> any:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "API backend requires the openai library. "
            "Install with: pip install openai"
        )
    client = AsyncOpenAI(api_key=api_key)
    print(f"Created API backend for model {model_name}")
    return client


def generate_response(
    client: any,
    api_params: dict,
) -> 

# The problem is that API models have different calling conventions

# But HuggingFace and Vllm models share the same conventions

# We may simply discard the general generate function for API backends


# Polymorphism can be implemented solely on Python side

# Rust side provides the backend object

# Rust only needs the outer interface

# Interface methods can be dispatched to Python implementations