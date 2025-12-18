import os
import json
import math
import asyncio
import torch


def language_abbreviation_to_name(abbreviation):
    """
    Map language abbreviation to full language name.
    """
    lang_map = {
        'en': 'English',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'zh_cn': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        # Add more mappings as needed
    }
    assert isinstance(abbreviation, str), "Language abbreviation must be a string"
    assert abbreviation in lang_map or len(abbreviation) > 2, f"Unknown language abbreviation: {abbreviation}"
    return lang_map.get(abbreviation, abbreviation)


async def collect_perplexity_local_async(entries, backend, model_interface, batch_size=8):
    """
    Calculate the perplexity of each answer entry using concurrent async requests.

    Perplexity is calculated by getting the average log probability of tokens in the answer
    given the question context. Lower perplexity indicates the model finds the answer more likely.

    Args:
        entries: List of individual answer entries, each containing:
            - 'index': int
            - 'question': str
            - 'answer': str
            - 'lang': str
            - 'is_correct': bool
            - 'subject': str
        backend: AsyncModelBackend instance (must be HuggingFace backend)
        model_interface: ModelInterface instance for model-specific behavior
        batch_size: Number of concurrent requests (default: 8)

    Returns:
        List of results to be written to file
    """

    tokenizer = backend.tokenizer
    model_name = getattr(backend, 'model_name', 'unknown')

    print(f"\nCalculating perplexities using local LLM")
    print(f"Concurrent requests: {batch_size}")
    print(f"Samples to process: {len(entries)}")

    # Process samples with concurrency control
    semaphore = asyncio.Semaphore(batch_size)
    results = []
    processed_count = 0

    async def process_single_entry(entry):
        """Process a single entry asynchronously."""
        nonlocal processed_count

        async with semaphore:
            try:
                language_name = language_abbreviation_to_name(entry['lang'])

                # Use the new interface to get logits for perplexity
                forward_result = await model_interface.forward_for_logits_async(
                    backend=backend,
                    question=entry['question'],
                    answer=entry['answer'],
                    language=language_name
                )

                # Calculate perplexity from forward result
                logits = forward_result.logits  # [seq_len, vocab_size]
                input_ids = forward_result.input_ids

                # Get answer tokens
                answer_tokens = tokenizer(entry['answer'], add_special_tokens=False).input_ids

                # Find answer position
                answer_start = model_interface.find_answer_start(
                    tokenizer, input_ids, answer_tokens
                )
                answer_end = answer_start + len(answer_tokens)

                # Shift logits and labels for perplexity calculation
                shift_logits = logits[:-1, :]  # All but last position
                shift_labels = torch.tensor(input_ids[1:])  # All but first position

                # Compute log probabilities
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                selected_log_probs = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)

                # Extract log probs for answer tokens (adjusting for shift)
                mask_start = answer_start - 1
                mask_end = answer_end - 1
                answer_log_probs = selected_log_probs[mask_start:mask_end]

                # Calculate perplexity
                if len(answer_log_probs) > 0:
                    avg_log_prob = answer_log_probs.mean().item()
                    perplexity = math.exp(-avg_log_prob)
                else:
                    perplexity = None

                output_result = {
                    'index': entry['index'],
                    'perplexity': perplexity,
                    'question': entry['question'],
                    'answer': entry['answer'],
                    'lang': entry['lang'],
                    'is_correct': entry['is_correct'],
                    'subject': entry.get('subject', ''),
                    'model': model_name,
                }

                return output_result

            except Exception as e:
                print(f"Error processing entry {entry['index']}: {e}")
                raise

    # Create all tasks
    tasks = [process_single_entry(entry) for entry in entries]

    # Process results as they complete
    for coro in asyncio.as_completed(tasks):
        result = await coro
        processed_count += 1
        results.append(result)

        if processed_count % 10 == 0 or processed_count == len(entries):
            print(f"  Processed {processed_count}/{len(entries)} samples")

    return results
