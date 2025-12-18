from typing import Any, List
from vllm import SamplingParams



def collect_perplexity_batch(
    entries: List[dict],
    model: Any,
    tokenizer: Any,
) -> List[dict]:
    """
    Collect raw logits and input_ids for a batch of entries using HuggingFace backend.

    Args:
        entries: List of entries, each containing 'question' and 'answer' fields
        model: HuggingFace model instance
        tokenizer: Tokenizer instance

    Returns:
        List of dicts containing 'logits' and 'input_ids' for each entry
    """
    import torch
    from src_py.utils import language_abbreviation_to_name

    results = []

    for entry in entries:
        question = entry['question']
        answer = entry['answer']
        lang = entry.get('lang', 'en')

        # Map language abbreviation to full name
        language_name = language_abbreviation_to_name(lang)

        # Build language-specific instructions (following qwen3_interface.py format)
        instruction = f"Please answer the question in {language_name} with a concise phrase instead of a complete sentence."

        # Combine question with instruction
        user_content = f"{question}\n\n{instruction}"

        # Build messages for chat template
        messages = [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        # Apply chat template to get the full formatted prompt
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Tokenize the formatted prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids[0]

        # Move input to model's device
        input_ids_tensor = input_ids.unsqueeze(0).to(model.device)

        # Get logits from model
        with torch.no_grad():
            outputs = model(input_ids_tensor)
            logits = outputs.logits[0].cpu()  # [seq_len, vocab_size], move to CPU

        # Store results with input_ids and logits
        results.append({
            'logits': logits,
            'input_ids': input_ids.tolist(),
            'answer': answer  # Keep answer for backward search
        })

    return results


async def collect_preference_local_async(
    question: str,
    answer1: str,
    answer2: str,
    engine: Any,
    tokenizer: Any,
) -> tuple[float, float]:
    """
    Collect preference between two answers using Llama 3.1 backend.

    Returns:
        Tuple of (logprob_1, logprob_2) where:
        - logprob_1: log probability of token "1"
        - logprob_2: log probability of token "2"
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are an impartial judge. The user is going to provide one question and two answers. "
                'If Answer 1 is better, respond with "1". '
                'If Answer 2 is better, respond with "2". '
                "Even if the answers are identical in correctness, try your best to choose a more favorable one. "
                "IMPORTANT: You SHOULD NOT judge an answer's quality based on its language.\n"
                'Only respond with "1" or "2", without any explanation.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Answer 1: {answer1}\n"
                f"Answer 2: {answer2}\n"
                "Which answer is better? Respond with '1' for Answer 1 or '2' for Answer 2."
            ),
        },
    ]

    # Convert chat messages to prompt text
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    sampling_params = SamplingParams(
        temperature=1.0,      # sampling at temperature 1.0
        max_tokens=100,         # only need one token
        stop=None,
        logprobs=10,          # get top 10 token log probabilities
    )

    # vLLM async generation (returns an async generator)
    request_id = f"qwen3_preference_{id(question)}"
    final_output = None
    async for output in engine.generate(prompt, sampling_params, request_id):
        # We only need the final result
        final_output = output

    if final_output is None:
        raise RuntimeError("vLLM generation returned no output")

    # Get the generated text for debugging
    generated_text = final_output.outputs[0].text if final_output.outputs[0].text else ""

    # Get the first token's logprobs
    if not final_output.outputs[0].logprobs or len(final_output.outputs[0].logprobs) == 0:
        raise RuntimeError(f"No logprobs returned from vLLM. Generated text: {repr(generated_text)}")

    first_token_logprobs = final_output.outputs[0].logprobs[0]

    # Get token IDs for "1" and "2"
    token_1_id = tokenizer.encode("1", add_special_tokens=False)[0]
    token_2_id = tokenizer.encode("2", add_special_tokens=False)[0]

    # Get the top tokens in the first position for debugging
    top_tokens_info = []
    for token_id, logprob_obj in sorted(first_token_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)[:5]:
        token_text = tokenizer.decode([token_id])
        top_tokens_info.append(f"ID {token_id} ({repr(token_text)}): {logprob_obj.logprob:.4f}")
    top_tokens_str = ", ".join(top_tokens_info)

    # Extract log probabilities for tokens "1" and "2"
    logprob_1 = None
    logprob_2 = None

    for token_id, logprob_obj in first_token_logprobs.items():
        if token_id == token_1_id:
            logprob_1 = logprob_obj.logprob
        elif token_id == token_2_id:
            logprob_2 = logprob_obj.logprob

    # Check if both tokens are in the top k
    if logprob_1 is None:
        raise ValueError(
            f"Token '1' (ID: {token_1_id}) not found in top-k logprobs. "
            f"Generated text: {repr(generated_text)}. "
            f"Top-5 first tokens: {top_tokens_str}"
        )
    if logprob_2 is None:
        raise ValueError(
            f"Token '2' (ID: {token_2_id}) not found in top-k logprobs. "
            f"Generated text: {repr(generated_text)}. "
            f"Top-5 first tokens: {top_tokens_str}"
        )

    return logprob_1, logprob_2
