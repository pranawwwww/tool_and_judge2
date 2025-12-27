"""
Common utilities for stacked bar chart generation.
Provides shared functionality for loading and processing result files.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------
# Translate + Noise modes
# -----------------------------
translate_modes = [
    "NT", # Not Translated
    "PAR", # Partially Translated
    "FT", # Fully Translated
    "PT", # Fully Translated + Prompt Translate
    "PRE", # Fully Translated + Pre-Translate
    "POST", # Fully Translated + Post-Translate
]

noise_modes = ["NO_NOISE", "PARAPHRASE", "SYNONYM"]

# Error categories (from ToolErrorCategory enum) - PascalCase format
error_categories = [
    "SyntaxError",
    "MiscError",
    "LanguageMismatchWrongValue",
    "LanguageMismatchRelevantButIncorrect",
    "LanguageMismatchExactlySameMeaning",
    "WrongValue",
    "RelevantButIncorrect",
    "ExactlySameMeaning",
    "OtherError",
]

# Color map for each error category (darker, uniform brightness)
category_colors = {
    "SyntaxError": "#b30000",  # Dark red
    "MiscError": "#b35900",  # Dark orange
    "LanguageMismatchWrongValue": "#5a1585", # Dark purple
    "LanguageMismatchRelevantButIncorrect": "#7a1aa0",  # Medium purple
    "LanguageMismatchExactlySameMeaning": "#9060b0",  # Light purple
    "WrongValue": "#b3b300",  # Dark yellow
    "RelevantButIncorrect": "#88b300",  # Dark yellow-green
    "ExactlySameMeaning": "#269900",  # Dark green
    "OtherError": "#707070",  # Dark gray
}

# Language name to tag mapping
language_tag_map = {
    "English": "en",
    "Chinese": "zh",
    "Hindi": "hi",
    "Igbo": "igbo"
}

# Category mapping from statistics file format to PascalCase
category_map = {
    "SYNTAX_ERROR": "SyntaxError",
    "MISC_ERROR": "MiscError",
    "LANGUAGE_MISMATCH_WRONG_VALUE": "LanguageMismatchWrongValue",
    "LANGUAGE_MISMATCH_RELEVANT_BUT_INCORRECT": "LanguageMismatchRelevantButIncorrect",
    "LANGUAGE_MISMATCH_EXACTLY_SAME_MEANING": "LanguageMismatchExactlySameMeaning",
    "WRONG_VALUE": "WrongValue",
    "RELEVANT_BUT_INCORRECT": "RelevantButIncorrect",
    "EXACTLY_SAME_MEANING": "ExactlySameMeaning",
    "OTHER_ERROR": "OtherError",
}


def pascal_to_readable(pascal_str: str) -> str:
    """
    Convert PascalCase string to lowercase with spaces.
    E.g., "LanguageMismatchExactlySameMeaning" -> "language mismatch exactly same meaning"
    """
    import re
    # Insert space before uppercase letters (except the first one)
    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', pascal_str)
    return spaced.lower()


def parse_filename_tags(filename: str) -> Tuple[str, str, str, str, str, str, str]:
    """
    Parse the filename (without .json extension) as a JSON array.

    Expected format: ["BFCL_v4_multiple", lang, trans_level, pre_trans, noise, prompt, post_trans]

    Returns:
        Tuple of (dataset_name, language_tag, translate_level_tag, pre_translate_tag,
                  noise_tag, prompt_translate_tag, post_translate_tag)
    """
    try:
        tags = json.loads(filename)
    except json.JSONDecodeError:
        raise ValueError(f"Cannot parse filename as JSON: '{filename}'")

    if len(tags) != 7:
        raise ValueError(f"Unexpected filename format (expected 7 elements, got {len(tags)})")

    return tuple(tags)


def map_noise_tag_to_mode(noise_tag: str) -> str:
    """Map noise_tag to noise_mode."""
    if noise_tag == "nonoise":
        return "NO_NOISE"
    elif noise_tag == "para":
        return "PARAPHRASE"
    elif noise_tag == "syno":
        return "SYNONYM"
    else:
        raise ValueError(f"Unknown noise tag '{noise_tag}'")


def map_tags_to_translate_mode(file_language_tag: str, translate_level_tag: str,
                                pre_translate_tag: str, prompt_translate_tag: str,
                                post_translate_tag: str) -> str:
    """
    Map combination of tags to translate_mode.

    Returns:
        One of: "NT", "PAR", "FT", "PT", "PRE", "POST"
    """
    if file_language_tag == "en" and translate_level_tag == "na":
        return "NT"
    elif file_language_tag in ["zh", "hi", "igbo"] and translate_level_tag == "parttrans":
        return "PAR"
    elif file_language_tag in ["zh", "hi", "igbo"] and translate_level_tag == "fulltrans":
        if (pre_translate_tag == "nopretrans" and prompt_translate_tag == "noprompt" and
            post_translate_tag == "noposttrans"):
            return "FT"
        elif (pre_translate_tag == "nopretrans" and prompt_translate_tag == "prompt" and
              post_translate_tag == "noposttrans"):
            return "PT"
        elif (pre_translate_tag == "pretrans" and prompt_translate_tag == "noprompt" and
              post_translate_tag == "noposttrans"):
            return "PRE"
        elif (pre_translate_tag == "nopretrans" and prompt_translate_tag == "noprompt" and
              post_translate_tag == "posttrans"):
            return "POST"
        else:
            raise ValueError(f"Unknown tag combination")
    else:
        raise ValueError(f"Unknown language/translate_level combination")


def load_model_statistics(model_name: str, result_dir: str, language: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Load error statistics for a specific model and language.

    Args:
        model_name: The model directory name (e.g., "gpt-5", "gpt-5-mini", "gpt-5-nano")
        result_dir: Directory containing the result files (default: "tool/result")
        language: The language name (e.g., "Chinese", "Hindi", "Igbo")

    Returns:
        Nested dictionary: dict[translate_mode][noise_mode][category] = count
    """
    # Get the language tag for filtering
    language_tag = language_tag_map.get(language)
    if language_tag is None:
        raise ValueError(f"Unknown language '{language}'. Valid options: {', '.join(language_tag_map.keys())}")

    # Initialize data structure
    data_dict = {}
    for tm in translate_modes:
        data_dict[tm] = {}
        for nm in noise_modes:
            data_dict[tm][nm] = {cat: 0 for cat in error_categories}

    # Path to model's statistics directory
    model_categorize_dir = Path(result_dir) / model_name / "statistics"

    if not model_categorize_dir.exists():
        raise ValueError(f"Model directory '{model_categorize_dir}' does not exist")

    # Parse statistics files
    for score_file in model_categorize_dir.glob("*.json"):
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                category_counts = data.get("category_counts")

                if category_counts is None:
                    print(f"Warning: No 'category_counts' field in {score_file.name}")
                    continue

                # Extract translate and noise modes from filename
                filename = score_file.stem  # Remove .json extension

                try:
                    (dataset_name, file_language_tag, translate_level_tag, pre_translate_tag,
                     noise_tag, prompt_translate_tag, post_translate_tag) = parse_filename_tags(filename)
                except ValueError as e:
                    print(f"Warning: {e}")
                    continue

                # Skip files that don't match the selected language
                # Exception: Always include English "en" as it represents the NT (Not Translated) baseline
                if file_language_tag != language_tag and file_language_tag != "en":
                    continue

                # Map tags to modes
                try:
                    noise_mode = map_noise_tag_to_mode(noise_tag)
                    translate_mode = map_tags_to_translate_mode(
                        file_language_tag, translate_level_tag,
                        pre_translate_tag, prompt_translate_tag, post_translate_tag
                    )
                except ValueError as e:
                    print(f"Warning: {e} in {score_file.name}")
                    continue

                # Store the category counts
                for stats_category, count in category_counts.items():
                    pascal_category = category_map.get(stats_category)
                    if pascal_category and pascal_category in error_categories:
                        data_dict[translate_mode][noise_mode][pascal_category] = count

                print(f"Loaded {score_file.name}: {translate_mode} + {noise_mode}")

        except Exception as e:
            print(f"Error reading {score_file.name}: {e}")

    return data_dict


def load_multi_model_statistics(model_names: List[str], result_dir: str, language: str,
                                 translate_mode: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Load error statistics for multiple models for a specific language and translate mode.

    Args:
        model_names: List of model directory names
        result_dir: Directory containing the result files
        language: The language name (e.g., "Chinese", "Hindi", "Igbo")
        translate_mode: The translate mode to filter by (e.g., "FT", "PT")

    Returns:
        Nested dictionary: dict[model_name][noise_mode][category] = count
    """
    # Get the language tag for filtering
    language_tag = language_tag_map.get(language)
    if language_tag is None:
        raise ValueError(f"Unknown language '{language}'. Valid options: {', '.join(language_tag_map.keys())}")

    # Initialize data structure
    data_dict = {}
    for model_name in model_names:
        data_dict[model_name] = {}
        for nm in noise_modes:
            data_dict[model_name][nm] = {cat: 0 for cat in error_categories}

    # Load data for each model
    for model_name in model_names:
        model_categorize_dir = Path(result_dir) / model_name / "statistics"

        if not model_categorize_dir.exists():
            print(f"Warning: Model directory '{model_categorize_dir}' does not exist, skipping")
            continue

        # Parse statistics files
        for score_file in model_categorize_dir.glob("*.json"):
            try:
                with open(score_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    category_counts = data.get("category_counts")

                    if category_counts is None:
                        print(f"Warning: No 'category_counts' field in {score_file.name}")
                        continue

                    # Extract translate and noise modes from filename
                    filename = score_file.stem  # Remove .json extension

                    try:
                        (dataset_name, file_language_tag, translate_level_tag, pre_translate_tag,
                         noise_tag, prompt_translate_tag, post_translate_tag) = parse_filename_tags(filename)
                    except ValueError as e:
                        print(f"Warning: {e}")
                        continue

                    # Skip files that don't match the selected language
                    # Exception: Always include English "en" as it represents the NT (Not Translated) baseline
                    if file_language_tag != language_tag and file_language_tag != "en":
                        continue

                    # Map tags to modes
                    try:
                        noise_mode = map_noise_tag_to_mode(noise_tag)
                        file_translate_mode = map_tags_to_translate_mode(
                            file_language_tag, translate_level_tag,
                            pre_translate_tag, prompt_translate_tag, post_translate_tag
                        )
                    except ValueError as e:
                        print(f"Warning: {e} in {score_file.name}")
                        continue

                    # Skip if not the desired translate mode
                    if file_translate_mode != translate_mode:
                        continue

                    # Store the category counts
                    for stats_category, count in category_counts.items():
                        pascal_category = category_map.get(stats_category)
                        if pascal_category and pascal_category in error_categories:
                            data_dict[model_name][noise_mode][pascal_category] = count

                    print(f"Loaded {score_file.name}: {model_name} + {noise_mode}")

            except Exception as e:
                print(f"Error reading {score_file.name}: {e}")

    return data_dict
