"""
HuggingFace backend for perplexity calculation using local models.
"""


def create_huggingface_backend(model_name: str, batch_size: int):
    """
    Create a HuggingFace backend for local model inference with batching support.

    Args:
        model_name: HuggingFace model name or path
        batch_size: Batch size for inference

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import os

    print(f"Creating HuggingFace backend for model {model_name} with batch size {batch_size}...")
    use_auth_token = os.environ["HF_TOKEN"]
    assert use_auth_token is not None, "HuggingFace token not found in environment variable HF_TOKEN."
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)

    # Set padding token if not set (needed for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with automatic device mapping
    print(f"Loading model from HuggingFace...", flush=True)

    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory available: {gpu_mem_gb:.2f} GB", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=use_auth_token,
        max_memory={0: "40GB"},  # Limit to prevent offloading to CPU
    )
    print(f"Model loaded, setting to eval mode...", flush=True)

    model.eval()
    print(f"Model in eval mode", flush=True)

    # Print device map to verify model placement
    if hasattr(model, 'hf_device_map'):
        print(f"Model device map: {model.hf_device_map}", flush=True)
        # Check if any layers are on CPU
        cpu_layers = [k for k, v in model.hf_device_map.items() if 'cpu' in str(v).lower()]
        if cpu_layers:
            print(f"WARNING: {len(cpu_layers)} layers are on CPU, which will cause slow inference!", flush=True)
    else:
        print(f"Model device: {model.device if hasattr(model, 'device') else 'unknown'}", flush=True)

    print(f"HuggingFace backend created successfully for {model_name}", flush=True)

    return (model, tokenizer)