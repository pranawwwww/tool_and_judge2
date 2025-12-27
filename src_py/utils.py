
from typing import Any
import importlib
import importlib.util
import os


import json
def load_config_from_file(config_file_path: str, config_var_name: str):
    """
    Load the 'config' list from a specified Python file.

    Args:
        config_file_path: Path to the Python file containing config

    Returns:
        The config list from the specified file
    """
    # Convert to absolute path if relative
    config_file_path = os.path.abspath(config_file_path)

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("custom_config", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if not hasattr(config_module, config_var_name):
        raise AttributeError(f"Config file {config_file_path} does not contain a '{config_var_name}' variable")

    return getattr(config_module, config_var_name)

def load_json_lines_from_file(file_path) -> list[Any]:
    """Load json lines from a JSONL file."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def combine_entries_to_pairs(entries1, entries2, lang1, lang2):
    """
    Combine two lists of entries into pairs by matching indices.

    Args:
        entries1: List of entries for answer1 (lang1)
        entries2: List of entries for answer2 (lang2)
        lang1: Language code for answer1
        lang2: Language code for answer2

    Returns:
        List of pairs with structure:
        {
            'index': int,
            'question': str,
            'answer1': str,
            'answer2': str,
            'lang1': str,
            'lang2': str,
            'is_correct1': bool,
            'is_correct2': bool,
            'subject': str,
        }
    """
    # Index entries by their index field
    entries1_by_index = {e['index']: e for e in entries1}
    entries2_by_index = {e['index']: e for e in entries2}

    # Find common indices
    common_indices = set(entries1_by_index.keys()) & set(entries2_by_index.keys())

    pairs = []
    for idx in sorted(common_indices):
        e1 = entries1_by_index[idx]
        e2 = entries2_by_index[idx]

        pair = {
            'index': idx,
            'question': e1['question'],
            'answer1': e1['answer'],
            'answer2': e2['answer'],
            'lang1': lang1,
            'lang2': lang2,
            'is_correct1': e1['is_correct'],
            'is_correct2': e2['is_correct'],
            'subject': e1.get('subject', ''),
        }
        pairs.append(pair)

    return pairs

def get_model_directory_safe_name(model_name: str) -> str:
    """Convert model name to a filesystem-safe directory name."""
    return model_name.replace("/", "-").replace(":", "-")

def language_abbreviation_to_name(abbreviation: str) -> str:
    """
    Map language abbreviation to full language name.

    Args:
        abbreviation: Language code (e.g., 'en', 'zh_cn', 'fr')

    Returns:
        Full language name (e.g., 'English', 'Chinese', 'French')
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
    }
    return lang_map.get(abbreviation.lower(), abbreviation)




def calculate_perplexity_from_logits(logits, input_ids, answer, tokenizer):
    """
    Calculate perplexity for an answer using backward search to locate answer tokens.

    Args:
        logits: torch.Tensor of shape [seq_len, vocab_size]
        input_ids: List of token IDs
        answer: str, the answer text
        tokenizer: Tokenizer instance

    Returns:
        float: perplexity value
    """
    import torch
    import math

    # Tokenize the answer to get its token sequence
    answer_tokens = tokenizer(answer, add_special_tokens=False).input_ids

    # Search backwards for the answer token sequence
    answer_start = None
    for i in range(len(input_ids) - len(answer_tokens), -1, -1):
        if input_ids[i:i+len(answer_tokens)] == answer_tokens:
            answer_start = i
            break

    if answer_start is None:
        raise ValueError(f"Could not find answer tokens in input_ids by backward search")

    answer_end = answer_start + len(answer_tokens)

    # Shift logits and labels for next-token prediction
    shift_logits = logits[:-1, :]  # All but last position
    shift_labels = torch.tensor(input_ids[1:])  # All but first position

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather log probs for the actual next tokens
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
        perplexity = float('inf')

    return perplexity