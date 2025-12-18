




import os


def create_vllm_backend(model_name: str, num_gpus: int = 1):
    """
    Create a vLLM backend for local model inference.

    Args:
        model_name: HuggingFace model name or path
        num_gpus: Number of GPUs to use for tensor parallelism

    Returns:
        Tuple of (AsyncLLMEngine, tokenizer)
    """
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from transformers import AutoTokenizer

    print(f"Creating vLLM backend for model {model_name} with {num_gpus} GPUs...")
    use_auth_token = os.environ["HF_TOKEN"]
    assert use_auth_token is not None, "HuggingFace token not found in environment variable HF_TOKEN."
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)

    # Create engine args
    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        enable_lora=False,
        max_model_len=5000,
    )

    # Initialize async engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    print(f"vLLM backend created successfully for {model_name}")

    return (engine, tokenizer)
