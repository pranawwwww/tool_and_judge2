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

    print(f"Creating HuggingFace backend for model {model_name} with batch size {batch_size}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token if not set (needed for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with automatic device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.eval()

    print(f"HuggingFace backend created successfully for {model_name}")

    return (model, tokenizer)