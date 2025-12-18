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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=use_auth_token,
    )

    model.eval()

    print(f"HuggingFace backend created successfully for {model_name}")

    return (model, tokenizer)