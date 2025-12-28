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
    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token is not None, "HuggingFace token not found in environment variable HF_TOKEN."
    # Load tokenizer (use 'token' parameter - 'use_auth_token' is deprecated)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Set padding token if not set (needed for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with automatic device mapping
    print(f"Loading model from HuggingFace...", flush=True)

    # Check available GPU memory and build max_memory dict
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs", flush=True)

        # Build max_memory dict for all available GPUs
        # Reserve some memory for activations and leave headroom
        max_memory = {}
        for i in range(num_gpus):
            gpu_mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            # Use 90% of available memory per GPU
            usable_mem_gb = int(gpu_mem_gb * 0.9)
            max_memory[i] = f"{usable_mem_gb}GB"
            print(f"GPU {i}: {gpu_mem_gb:.2f} GB available, using {usable_mem_gb} GB", flush=True)
    else:
        max_memory = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_token,
        max_memory=max_memory,
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