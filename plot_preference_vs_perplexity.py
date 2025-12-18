#!/usr/bin/env python3
"""
Generate scatter plots comparing preference logprob_signed_difference vs perplexity difference.

For each model and pair of languages, creates 4 plots corresponding to:
- lang1 correct, lang2 correct
- lang1 correct, lang2 incorrect
- lang1 incorrect, lang2 correct
- lang1 incorrect, lang2 incorrect

Each point plots:
- X-axis: perplexity difference (perplexity_lang1 - perplexity_lang2)
- Y-axis: preference logprob_signed_difference
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_jsonl(file_path: str) -> List[dict]:
    """Load JSON lines file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_perplexity_value(entry: dict) -> Optional[float]:
    """Extract perplexity value from Result type."""
    perplexity = entry.get('perplexity')
    if isinstance(perplexity, dict):
        if 'Ok' in perplexity:
            return perplexity['Ok']
    return None


def get_preference_value(entry: dict) -> Optional[float]:
    """Extract logprob_signed_difference from Result type."""
    preference = entry.get('preference')
    if isinstance(preference, dict):
        if 'Ok' in preference:
            return preference['Ok'].get('logprob_signed_difference')
    return None


def load_perplexity_data(model_name: str, lang: str) -> Dict[int, Dict[bool, float]]:
    """
    Load perplexity data for a given model and language.

    Returns:
        Dict mapping index -> {is_correct: perplexity_value}
    """
    base_path = Path(f"judge/result/{model_name}/perplexity")

    perplexity_map = {}

    # Load correct answers
    correct_file = base_path / f"{lang}_correct.jsonl"
    if correct_file.exists():
        correct_data = load_jsonl(str(correct_file))
        for entry in correct_data:
            idx = entry['index']
            perplexity = get_perplexity_value(entry)
            if perplexity is not None:
                if idx not in perplexity_map:
                    perplexity_map[idx] = {}
                perplexity_map[idx][True] = perplexity

    # Load incorrect answers
    incorrect_file = base_path / f"{lang}_incorrect.jsonl"
    if incorrect_file.exists():
        incorrect_data = load_jsonl(str(incorrect_file))
        for entry in incorrect_data:
            idx = entry['index']
            perplexity = get_perplexity_value(entry)
            if perplexity is not None:
                if idx not in perplexity_map:
                    perplexity_map[idx] = {}
                perplexity_map[idx][False] = perplexity

    return perplexity_map


def load_preference_data(model_name: str, lang1: str, lang2: str) -> Dict[Tuple[bool, bool], List[dict]]:
    """
    Load preference data for a given model and language pair.

    Returns:
        Dict mapping (is_correct1, is_correct2) -> list of preference entries
    """
    base_path = Path(f"judge/result/{model_name}/preference")

    preference_map = {
        (True, True): [],
        (True, False): [],
        (False, True): [],
        (False, False): []
    }

    # Map of files to correctness tuples
    files_map = [
        (f"{lang1}_correct_{lang2}_correct.jsonl", (True, True)),
        (f"{lang1}_correct_{lang2}_incorrect.jsonl", (True, False)),
        (f"{lang1}_incorrect_{lang2}_correct.jsonl", (False, True)),
        (f"{lang1}_incorrect_{lang2}_incorrect.jsonl", (False, False)),
    ]

    for filename, key in files_map:
        file_path = base_path / filename
        if file_path.exists():
            data = load_jsonl(str(file_path))
            preference_map[key] = data

    return preference_map


def filter_outliers_percentile(
    perplexity_diffs: List[float],
    preference_values: List[float],
    keep_percentage: float = 0.90
) -> Tuple[List[float], List[float], int]:
    """
    Filter outliers by removing extreme values based on percentiles.

    For both perplexity differences and preference values, keep only the middle
    keep_percentage of values (e.g., 90% means remove top 5% and bottom 5%).

    Args:
        perplexity_diffs: List of perplexity difference values
        preference_values: List of preference values
        keep_percentage: Percentage of data to keep (default 0.90 for 90%)

    Returns:
        Tuple of (filtered_perplexity_diffs, filtered_preference_values, num_outliers_removed)
    """
    if len(perplexity_diffs) == 0:
        return [], [], 0

    perp_arr = np.array(perplexity_diffs)
    pref_arr = np.array(preference_values)

    # Calculate percentiles for trimming
    # E.g., for 90%, keep between 5th and 95th percentile
    lower_percentile = (1 - keep_percentage) / 2 * 100
    upper_percentile = (1 + keep_percentage) / 2 * 100

    # Calculate bounds for perplexity differences
    perp_lower = np.percentile(perp_arr, lower_percentile)
    perp_upper = np.percentile(perp_arr, upper_percentile)

    # Calculate bounds for preference values
    pref_lower = np.percentile(pref_arr, lower_percentile)
    pref_upper = np.percentile(pref_arr, upper_percentile)

    # Create masks for points within bounds
    perp_mask = (perp_arr >= perp_lower) & (perp_arr <= perp_upper)
    pref_mask = (pref_arr >= pref_lower) & (pref_arr <= pref_upper)

    # Keep points that are within bounds in BOTH dimensions
    combined_mask = perp_mask & pref_mask

    # Filter the data
    filtered_perp = perp_arr[combined_mask].tolist()
    filtered_pref = pref_arr[combined_mask].tolist()

    num_removed = len(perplexity_diffs) - len(filtered_perp)

    return filtered_perp, filtered_pref, num_removed


def create_scatter_plot(
    perplexity_diffs: List[float],
    preference_values: List[float],
    lang1: str,
    lang2: str,
    is_correct1: bool,
    is_correct2: bool,
    output_path: str,
    num_outliers_removed: int = 0,
    total_points: int = None
):
    """Create and save a scatter plot."""
    plt.figure(figsize=(10, 8))

    # Create scatter plot
    plt.scatter(perplexity_diffs, preference_values, alpha=0.5, s=20)

    # Add reference lines
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, linewidth=1)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3, linewidth=1)

    # Labels and title
    plt.xlabel(f'Perplexity Difference ({lang1} - {lang2})', fontsize=12)
    plt.ylabel('Preference Log-Prob Signed Difference', fontsize=12)

    correct1_str = "correct" if is_correct1 else "incorrect"
    correct2_str = "correct" if is_correct2 else "incorrect"

    if total_points is not None and num_outliers_removed > 0:
        title = f'{lang1} {correct1_str} vs {lang2} {correct2_str}\n(n={len(perplexity_diffs)} after filtering, {num_outliers_removed} outliers removed from {total_points})'
    else:
        title = f'{lang1} {correct1_str} vs {lang2} {correct2_str}\n(n={len(perplexity_diffs)})'
    plt.title(title, fontsize=14)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add correlation coefficient
    if len(perplexity_diffs) > 0:
        corr = np.corrcoef(perplexity_diffs, preference_values)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate scatter plots of preference vs perplexity for a model'
    )
    parser.add_argument('model_name', type=str,
                       help='Model name (e.g., Qwen-Qwen3-8B)')
    parser.add_argument('lang1', type=str,
                       help='First language code (e.g., en)')
    parser.add_argument('lang2', type=str,
                       help='Second language code (e.g., zh_cn)')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--filter-outliers', action='store_true',
                       help='Filter outliers using percentile-based approach')
    parser.add_argument('--keep-percentage', type=float, default=0.90,
                       help='Percentage of data to keep after filtering outliers (default: 0.90 for 90%%)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for model: {args.model_name}")
    print(f"Languages: {args.lang1} vs {args.lang2}")
    if args.filter_outliers:
        print(f"Outlier filtering enabled: keeping {args.keep_percentage * 100:.0f}% of data")

    # Load perplexity data for both languages
    print(f"Loading perplexity data for {args.lang1}...")
    perplexity_lang1 = load_perplexity_data(args.model_name, args.lang1)
    print(f"  Found {len(perplexity_lang1)} indices with perplexity data")

    print(f"Loading perplexity data for {args.lang2}...")
    perplexity_lang2 = load_perplexity_data(args.model_name, args.lang2)
    print(f"  Found {len(perplexity_lang2)} indices with perplexity data")

    # Load preference data
    print(f"Loading preference data...")
    preference_data = load_preference_data(args.model_name, args.lang1, args.lang2)

    # Process each category
    for (is_correct1, is_correct2), entries in preference_data.items():
        print(f"\nProcessing {args.lang1} {'correct' if is_correct1 else 'incorrect'} vs "
              f"{args.lang2} {'correct' if is_correct2 else 'incorrect'}...")
        print(f"  Total preference entries: {len(entries)}")

        perplexity_diffs = []
        preference_values = []

        for entry in entries:
            idx = entry['index']

            # Get preference value
            pref_val = get_preference_value(entry)
            if pref_val is None:
                continue

            # Get perplexity values for both languages with correct/incorrect answers
            if idx not in perplexity_lang1 or idx not in perplexity_lang2:
                continue

            if is_correct1 not in perplexity_lang1[idx] or is_correct2 not in perplexity_lang2[idx]:
                continue

            perp1 = perplexity_lang1[idx][is_correct1]
            perp2 = perplexity_lang2[idx][is_correct2]

            # Calculate difference
            perp_diff = perp1 - perp2

            perplexity_diffs.append(perp_diff)
            preference_values.append(pref_val)

        print(f"  Valid data points: {len(perplexity_diffs)}")

        if len(perplexity_diffs) > 0:
            # Apply outlier filtering if requested
            num_outliers_removed = 0
            total_points = len(perplexity_diffs)

            if args.filter_outliers:
                original_count = len(perplexity_diffs)
                perplexity_diffs, preference_values, num_outliers_removed = filter_outliers_percentile(
                    perplexity_diffs,
                    preference_values,
                    keep_percentage=args.keep_percentage
                )
                print(f"  After outlier filtering: {len(perplexity_diffs)} points "
                      f"({num_outliers_removed} outliers removed, "
                      f"{100 * num_outliers_removed / original_count:.1f}%)")

            # Create output filename
            correct1_str = "correct" if is_correct1 else "incorrect"
            correct2_str = "correct" if is_correct2 else "incorrect"
            filename_suffix = "_filtered" if args.filter_outliers else ""
            output_filename = f"{args.model_name}_{args.lang1}_{correct1_str}_vs_{args.lang2}_{correct2_str}{filename_suffix}.png"
            output_path = output_dir / output_filename

            # Create plot
            create_scatter_plot(
                perplexity_diffs,
                preference_values,
                args.lang1,
                args.lang2,
                is_correct1,
                is_correct2,
                str(output_path),
                num_outliers_removed=num_outliers_removed,
                total_points=total_points if args.filter_outliers else None
            )
        else:
            print(f"  Warning: No valid data points found for this category")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
