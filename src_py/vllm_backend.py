import os
from codebase_rs import LocalModel

def create_vllm_backend(local_model: LocalModel,
        num_gpus: int = 1):
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
    model_name = local_model.to_string()
    print(f"Creating vLLM backend for model {model_name} with {num_gpus} GPUs...")
    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token is not None, "HuggingFace token not found in environment variable HF_TOKEN."
    # Load tokenizer (use 'token' parameter - 'use_auth_token' is deprecated)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

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

    return engine, tokenizer
