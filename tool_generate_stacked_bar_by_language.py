import matplotlib
matplotlib.use("Agg")  # HPC-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from typing import Dict, List

from tool_stacked_bar_common import (
    translate_modes,
    noise_modes,
    error_categories,
    category_colors,
    pascal_to_readable,
    language_tag_map,
    parse_filename_tags,
    map_noise_tag_to_mode,
    map_tags_to_translate_mode,
    category_map,
)


def load_multi_language_statistics(model_name: str, result_dir: str, languages: List[str],
                                     translate_mode: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Load error statistics for multiple languages for a specific model and translate mode.

    Args:
        model_name: The model directory name (e.g., "gpt-5", "gpt-5-mini", "gpt-5-nano")
        result_dir: Directory containing the result files
        languages: List of language names (e.g., ["Chinese", "Hindi", "Igbo"])
        translate_mode: The translate mode to filter by (e.g., "FT", "PT")

    Returns:
        Nested dictionary: dict[language][noise_mode][category] = count
    """
    # Get the language tags for filtering
    language_tags = {}
    for language in languages:
        language_tag = language_tag_map.get(language)
        if language_tag is None:
            raise ValueError(f"Unknown language '{language}'. Valid options: {', '.join(language_tag_map.keys())}")
        language_tags[language] = language_tag

    # Initialize data structure
    data_dict = {}
    for language in languages:
        data_dict[language] = {}
        for nm in noise_modes:
            data_dict[language][nm] = {cat: 0 for cat in error_categories}

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

                # Find which language this file belongs to
                matching_language = None
                for language, lang_tag in language_tags.items():
                    if file_language_tag == lang_tag:
                        matching_language = language
                        break

                # Skip files that don't match any selected language
                if matching_language is None:
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
                        data_dict[matching_language][noise_mode][pascal_category] = count

                print(f"Loaded {score_file.name}: {matching_language} + {noise_mode}")

        except Exception as e:
            print(f"Error reading {score_file.name}: {e}")

    return data_dict


def generate_stacked_bar_chart_by_language(model_name: str, output_dir: str,
                                             result_dir: str, translate_mode: str,
                                             max_height: float = None) -> None:
    """
    Generate a stacked bar chart for a single model comparing languages showing error type distributions.
    Horizontal axis shows language x noise mode combinations grouped by language.
    Always uses Chinese, Hindi, and Igbo as languages.

    Args:
        model_name: Model directory name (e.g., "gpt-5", "gpt-5-mini", "gpt-5-nano")
        output_dir: Directory to save the chart image
        result_dir: Directory containing the result files (default: "tool/result")
        translate_mode: The translate mode to filter by (e.g., "FT", "PT", "PRE", "POST")
        max_height: Maximum height of the vertical axis (default: None, auto-calculated from data)
    """

    # Fixed languages list
    languages = ["Chinese", "Hindi", "Igbo"]

    # Load data using common module
    try:
        data_dict = load_multi_language_statistics(model_name, result_dir, languages, translate_mode)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Prepare data for plotting - show all language x noise mode combinations grouped by language
    bar_labels = []
    bar_data = []
    bar_positions = []
    bar_widths = []  # Track custom widths for each bar
    pos = 0
    bar_spacing = 0.6  # Spacing between bars within a group
    group_spacing = 0.3  # Extra space between language groups

    # Map noise modes to short abbreviations
    noise_mode_abbrev = {
        "NO_NOISE": "NO",
        "PARAPHRASE": "PARA",
        "SYNONYM": "SYNO"
    }

    for language in languages:
        if language == "Igbo":
            # For Igbo: single wide bar spanning the width of three bars (only NO_NOISE)
            bar_labels.append("NO")  # Show "NO" sub-label
            category_counts = data_dict[language]["NO_NOISE"]
            bar_data.append(category_counts)
            # Calculate center position for a bar that spans three bar widths
            bar_center = pos + bar_spacing
            bar_positions.append(bar_center)
            # Width spans three bars and two gaps: 3 * 0.4 + 2 * (0.6 - 0.4) = 1.6
            bar_widths.append(1.6)
            pos += 3 * bar_spacing  # Move position as if we placed three bars
        else:
            # Normal case for Chinese and Hindi: three bars for three noise modes
            for nm in noise_modes:
                bar_labels.append(noise_mode_abbrev[nm])  # Use abbreviated noise mode name
                category_counts = data_dict[language][nm]
                bar_data.append(category_counts)
                bar_positions.append(pos)
                bar_widths.append(0.4)  # Standard bar width
                pos += bar_spacing  # Use smaller spacing between bars
        pos += group_spacing  # Add extra space after each language group

    title = f"Tool Calling Errors for {model_name} - {translate_mode}"
    output_name = f"stacked_bar_by_language_{model_name}_{translate_mode}_all_combined.png"

    # Create DataFrame for easier plotting
    df_data = []
    for counts in bar_data:
        df_data.append([counts[cat] for cat in error_categories])

    df = pd.DataFrame(df_data, index=bar_labels, columns=error_categories)

    # Check if we have any data
    if df.sum().sum() == 0:
        print(f"Error: No error data found for the specified model and configuration")
        return

    # Print summary
    print(f"\nError distribution for {model_name} - {translate_mode}:")
    print(df)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert counts to rates by dividing by 200
    df_rate = df / 200.0

    # Create stacked bars with custom positions
    x_positions = bar_positions
    bottom = np.zeros(len(bar_positions))

    for category in error_categories:
        values = df_rate[category].values
        # Convert category name to readable format for legend
        readable_label = pascal_to_readable(category)
        # Use custom bar widths for each bar
        ax.bar(x_positions, values, label=readable_label, bottom=bottom,
               color=category_colors[category], edgecolor='white', linewidth=0.5, width=bar_widths)
        bottom += values

    # Calculate totals for each bar (as rates)
    totals = df_rate.sum(axis=1).values

    # Calculate y-axis max height (ceiling to nearest 0.1)
    if max_height is None:
        data_max = totals.max()
        max_height = np.ceil(data_max * 10) / 10  # Round up to nearest 0.1
        if max_height == data_max:  # If already at boundary, add 0.1
            max_height += 0.1

    # Add total numbers on top of each bar
    for i, total in enumerate(totals):
        if total > 0:  # Only annotate if there's data
            ax.text(x_positions[i], total, f'{total:.3f}',
                   ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Set y-axis range to the calculated max_height
    ax.set_ylim(0, max_height)

    # Customize plot
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Place legend outside plot area on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11)

    # Handle x-axis labels and ticks
    # Set x-tick positions and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)

    # Add language group labels as a second row below noise mode labels
    bar_spacing = 0.6
    group_spacing = 0.3
    pos_tracker = 0
    for i, language in enumerate(languages):
        if language == "Igbo":
            # For Igbo: center is at the single wide bar
            group_center = pos_tracker + bar_spacing
            pos_tracker += 3 * bar_spacing + group_spacing
        else:
            # Normal case: center of three bars
            group_center = pos_tracker + (len(noise_modes) - 1) * bar_spacing / 2
            pos_tracker += len(noise_modes) * bar_spacing + group_spacing
        ax.text(group_center, -max_height * 0.08, language,
               ha='center', va='top', fontsize=12, fontweight='bold')

    # Add "Language and Configuration" label below the group names (larger negative offset)
    # Calculate the center of all bars for proper centering
    overall_center = (x_positions[0] + x_positions[-1]) / 2
    ax.text(overall_center, -max_height * 0.12, 'Language and Configuration',
           ha='center', va='top', fontsize=12)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved stacked bar chart to {output_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate stacked bar charts for a model comparing languages showing error type distributions. "
                    "Automatically uses Chinese, Hindi, and Igbo as languages."
    )
    parser.add_argument(
        "model",
        help="Model name (e.g., gpt-5, gpt-5-mini, gpt-5-nano)"
    )
    parser.add_argument(
        "translate_mode",
        choices=translate_modes,
        help="Translate mode to filter by (e.g., FT, PT, PRE, POST)"
    )
    parser.add_argument(
        "--output-dir",
        default="tool/plots/stacked_bars_by_language",
        help="Directory to save chart images (default: tool/plots/stacked_bars_by_language)"
    )
    parser.add_argument(
        "--result-dir",
        default="tool/result",
        help="Directory containing the result files (default: tool/result)"
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=None,
        help="Maximum height of the vertical axis (default: auto-calculated from data, rounded up to nearest 0.1)"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Generating stacked bar chart for model: {args.model}")
    print(f"Languages: Chinese, Hindi, Igbo, Translate Mode: {args.translate_mode}")
    print(f"{'='*60}")

    # Generate single combined chart with all language x noise mode combinations
    generate_stacked_bar_chart_by_language(
        args.model,
        args.output_dir,
        args.result_dir,
        args.translate_mode,
        max_height=args.max_height
    )
