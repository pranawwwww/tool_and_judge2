import matplotlib
matplotlib.use("Agg")  # HPC-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

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


def pascal_to_readable(pascal_str: str) -> str:
    """
    Convert PascalCase string to lowercase with spaces.
    E.g., "LanguageMismatchExactlySameMeaning" -> "language mismatch exactly same meaning"
    """
    import re
    # Insert space before uppercase letters (except the first one)
    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', pascal_str)
    return spaced.lower()


def generate_stacked_bar_chart(model_name: str, output_dir: str, result_dir: str,
                                language: str,
                                selected_translate_mode: str = None,
                                selected_noise_mode: str = None,
                                max_height: float = 0.5,
                                show_all_combined: bool = False) -> None:
    """
    Generate a stacked bar chart for a given model showing error type distributions.

    Args:
        model_name: The model directory name (e.g., "gpt-5", "gpt-5-mini", "gpt-5-nano")
        output_dir: Directory to save the chart image (default: current directory)
        result_dir: Directory containing the categorize_score files (default: "tool/result/categorize_score")
        language: The language name for the plot title (e.g., "Chinese", "Hindi", "Igbo")
        selected_translate_mode: If specified, only show bars for this translate mode across noise modes
        selected_noise_mode: If specified, only show bars for this noise mode across translate modes
        max_height: Maximum height of the vertical axis (default: 0.5, range: 0.0-1.0)
        show_all_combined: If True, show all 18 combinations in a single plot with grouping
    """

    # Initialize data structure: dict[translate_mode][noise_mode][category] = count
    data_dict = {}
    for tm in translate_modes:
        data_dict[tm] = {}
        for nm in noise_modes:
            data_dict[tm][nm] = {cat: 0 for cat in error_categories}

    # Path to model's statistics directory
    model_categorize_dir = Path(result_dir) / model_name / "statistics"

    if not model_categorize_dir.exists():
        print(f"Error: Model directory '{model_categorize_dir}' does not exist")
        return

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

                # Parse the filename as a JSON array
                # Expected format: ["BFCL_v4_multiple", lang, trans_level, pre_trans, noise, prompt, post_trans]
                try:
                    tags = json.loads(filename)
                except json.JSONDecodeError:
                    print(f"Warning: Cannot parse filename as JSON: '{score_file.name}'")
                    continue

                # We expect 7 elements in this order:
                # 0: dataset name (e.g., "BFCL_v4_multiple")
                # 1: language_tag (en, zh, hi, igbo)
                # 2: translate_level_tag (na, parttrans, fulltrans)
                # 3: pre_translate_tag (pretrans, nopretrans)
                # 4: noise_tag (nonoise, syno, para)
                # 5: prompt_translate_tag (prompt, noprompt)
                # 6: post_translate_tag (posttrans, noposttrans)

                if len(tags) != 7:
                    print(f"Warning: Unexpected filename format '{score_file.name}' (expected 7 elements, got {len(tags)})")
                    continue

                language_tag = tags[1]
                translate_level_tag = tags[2]
                pre_translate_tag = tags[3]
                noise_tag = tags[4]
                prompt_translate_tag = tags[5]
                post_translate_tag = tags[6]

                # Map noise_tag to noise_mode
                if noise_tag == "nonoise":
                    noise_mode = "NO_NOISE"
                elif noise_tag == "para":
                    noise_mode = "PARAPHRASE"
                elif noise_tag == "syno":
                    noise_mode = "SYNONYM"
                else:
                    print(f"Warning: Unknown noise tag '{noise_tag}' in {score_file.name}")
                    continue

                # Map combination of tags to translate_mode (same logic as heatmap)
                if language_tag == "en" and translate_level_tag == "na":
                    translate_mode = "NT"
                elif language_tag in ["zh", "hi", "igbo"] and translate_level_tag == "parttrans":
                    translate_mode = "PAR"
                elif language_tag in ["zh", "hi", "igbo"] and translate_level_tag == "fulltrans":
                    if (pre_translate_tag == "nopretrans" and prompt_translate_tag == "noprompt" and
                        post_translate_tag == "noposttrans"):
                        translate_mode = "FT"
                    elif (pre_translate_tag == "nopretrans" and prompt_translate_tag == "prompt" and
                          post_translate_tag == "noposttrans"):
                        translate_mode = "PT"
                    elif (pre_translate_tag == "pretrans" and prompt_translate_tag == "noprompt" and
                          post_translate_tag == "noposttrans"):
                        translate_mode = "PRE"
                    elif (pre_translate_tag == "nopretrans" and prompt_translate_tag == "noprompt" and
                          post_translate_tag == "posttrans"):
                        translate_mode = "POST"
                    else:
                        print(f"Warning: Unknown tag combination in {score_file.name}")
                        continue
                else:
                    print(f"Error: Unknown language/translate_level combination in {score_file.name}")
                    exit(1)

                # Store the category counts
                # Map from statistics file format (UPPERCASE_WITH_UNDERSCORES) to PascalCase
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
                for stats_category, count in category_counts.items():
                    pascal_category = category_map.get(stats_category)
                    if pascal_category and pascal_category in error_categories:
                        data_dict[translate_mode][noise_mode][pascal_category] = count

                print(f"Loaded {score_file.name}: {translate_mode} + {noise_mode}")

        except Exception as e:
            print(f"Error reading {score_file.name}: {e}")

    # Prepare data for plotting
    if selected_translate_mode and selected_noise_mode:
        print("Error: Cannot specify both translate_mode and noise_mode")
        return
    elif show_all_combined:
        # Show all 18 combinations grouped by translate mode
        bar_labels = []
        bar_data = []
        bar_positions = []
        pos = 0
        bar_spacing = 0.6  # Spacing between bars within a group
        group_spacing = 0.3  # Extra space between translate mode groups

        # Map noise modes to short abbreviations
        noise_mode_abbrev = {
            "NO_NOISE": "NO",
            "PARAPHRASE": "PARA",
            "SYNONYM": "SYNO"
        }

        for tm in translate_modes:
            for nm in noise_modes:
                bar_labels.append(noise_mode_abbrev[nm])  # Use abbreviated noise mode name
                category_counts = data_dict[tm][nm]
                bar_data.append(category_counts)
                bar_positions.append(pos)
                pos += bar_spacing  # Use smaller spacing between bars
            pos += group_spacing  # Add extra space after each translate mode group

        title = f"Tool Calling Errors of {model_name} Under {language} Queries"
        output_name = f"stacked_bar_{model_name}_{language}_all_combined.png"
    elif selected_translate_mode:
        # Show bars for selected translate mode across noise modes
        bar_labels = noise_modes
        bar_data = []
        bar_positions = None
        for nm in noise_modes:
            category_counts = data_dict[selected_translate_mode][nm]
            bar_data.append(category_counts)
        title = f"Tool Calling Errors of {model_name} Under {language} Queries - {selected_translate_mode}"
        output_name = f"stacked_bar_{model_name}_{language}_{selected_translate_mode}.png"
    elif selected_noise_mode:
        # Show bars for selected noise mode across translate modes
        bar_labels = translate_modes
        bar_data = []
        bar_positions = None
        for tm in translate_modes:
            category_counts = data_dict[tm][selected_noise_mode]
            bar_data.append(category_counts)
        title = f"Tool Calling Error Rate of {model_name} Under {language} Queries - {selected_noise_mode}"
        output_name = f"stacked_bar_{model_name}_{language}_{selected_noise_mode}.png"
    else:
        # Default: show all combinations (might be too many bars)
        bar_labels = []
        bar_data = []
        bar_positions = None
        for tm in translate_modes:
            for nm in noise_modes:
                bar_labels.append(f"{tm}_{nm}")
                category_counts = data_dict[tm][nm]
                bar_data.append(category_counts)
        title = f"Tool Calling Error Rate of {model_name} Under {language} Queries - All Combinations"
        output_name = f"stacked_bar_{model_name}_{language}_all.png"

    # Create DataFrame for easier plotting
    df_data = []
    for counts in bar_data:
        df_data.append([counts[cat] for cat in error_categories])

    df = pd.DataFrame(df_data, index=bar_labels, columns=error_categories)

    # Check if we have any data
    if df.sum().sum() == 0:
        print(f"Error: No error data found for model '{model_name}'")
        return

    # Print summary
    print(f"\nError distribution for model '{model_name}':")
    print(df)

    # Plot stacked bar chart (wider if showing all combined)
    if show_all_combined:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Convert counts to rates by dividing by 200
    df_rate = df / 200.0

    # Create stacked bars with custom positions if specified
    if bar_positions is not None:
        x_positions = bar_positions
        bottom = np.zeros(len(bar_positions))
    else:
        x_positions = range(len(bar_labels))
        bottom = np.zeros(len(bar_labels))

    for category in error_categories:
        values = df_rate[category].values
        # Convert category name to readable format for legend
        readable_label = pascal_to_readable(category)
        # Use half width (0.4) for combined view, standard width (0.8) otherwise
        bar_width = 0.4 if show_all_combined else 0.8
        ax.bar(x_positions, values, label=readable_label, bottom=bottom,
               color=category_colors[category], edgecolor='white', linewidth=0.5, width=bar_width)
        bottom += values

    # Calculate totals for each bar (as rates)
    totals = df_rate.sum(axis=1).values

    # Add total numbers on top of each bar
    for i, total in enumerate(totals):
        if total > 0:  # Only annotate if there's data
            x_pos = x_positions[i] if bar_positions is not None else i
            ax.text(x_pos, total, f'{total:.3f}',
                   ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Set y-axis range to the specified max_height
    ax.set_ylim(0, max_height)

    # Customize plot
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Place legend outside plot area on the right with smaller font
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

    # Handle x-axis labels and ticks
    if show_all_combined:
        # Set x-tick positions and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)

        # Add translate mode group labels as a second row below noise mode labels
        # Position them closer to the axis (smaller negative offset)
        bar_spacing = 0.6
        group_spacing = 0.3
        for i, tm in enumerate(translate_modes):
            # Calculate the center position of each translate mode group
            group_center = i * (len(noise_modes) * bar_spacing + group_spacing) + (len(noise_modes) - 1) * bar_spacing / 2
            ax.text(group_center, -max_height * 0.08, tm,
                   ha='center', va='top', fontsize=10, fontweight='bold')

        # Add "Configuration" label below the group names (larger negative offset)
        # Calculate the center of all bars for proper centering
        overall_center = (x_positions[0] + x_positions[-1]) / 2
        ax.text(overall_center, -max_height * 0.12, 'Configuration',
               ha='center', va='top', fontsize=12)
    elif len(bar_labels) > 10:
        ax.set_xlabel('Configuration', fontsize=12)
        plt.xticks(rotation=45, ha='right')
    else:
        ax.set_xlabel('Configuration', fontsize=12)
        plt.xticks(rotation=0)

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
        description="Generate stacked bar charts showing error type distributions for specified models."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Model names to process (e.g., gpt-5 gpt-5-mini gpt-5-nano)"
    )
    parser.add_argument(
        "language",
        help="Language name for the plot title (e.g., Chinese, Hindi, Igbo)"
    )
    parser.add_argument(
        "--output-dir",
        default="tool/plots/stacked_bars",
        help="Directory to save chart images (default: current directory)"
    )
    parser.add_argument(
        "--result-dir",
        default="tool/result",
        help="Directory containing the result files (default: tool/result)"
    )
    parser.add_argument(
        "--translate-mode",
        choices=translate_modes,
        help="Only show bars for this translate mode across noise modes"
    )
    parser.add_argument(
        "--noise-mode",
        choices=noise_modes,
        help="Only show bars for this noise mode across translate modes"
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=0.5,
        help="Maximum height of the vertical axis (default: 0.5, range: 0.0-1.0)"
    )
    parser.add_argument(
        "--all-combined",
        action="store_true",
        help="Generate a single 18-column plot with all translate and noise modes grouped together"
    )

    args = parser.parse_args()

    for model in args.models:
        print(f"\n{'='*60}")
        print(f"Generating stacked bar charts for {model}")
        print(f"{'='*60}")

        if args.all_combined:
            # Generate single combined chart with all 18 combinations
            generate_stacked_bar_chart(
                model,
                args.output_dir,
                args.result_dir,
                args.language,
                max_height=args.max_height,
                show_all_combined=True
            )
        elif args.translate_mode or args.noise_mode:
            # Generate chart with specified filter
            generate_stacked_bar_chart(
                model,
                args.output_dir,
                args.result_dir,
                args.language,
                selected_translate_mode=args.translate_mode,
                selected_noise_mode=args.noise_mode,
                max_height=args.max_height
            )
        else:
            # Generate chart for NO_NOISE across all translate modes
            generate_stacked_bar_chart(
                model,
                args.output_dir,
                args.result_dir,
                args.language,
                selected_noise_mode="NO_NOISE",
                max_height=args.max_height
            )

            # Generate chart for PARAPHRASE across all translate modes
            generate_stacked_bar_chart(
                model,
                args.output_dir,
                args.result_dir,
                args.language,
                selected_noise_mode="PARAPHRASE",
                max_height=args.max_height
            )

            # Generate chart for SYNONYM across all translate modes
            generate_stacked_bar_chart(
                model,
                args.output_dir,
                args.result_dir,
                args.language,
                selected_noise_mode="SYNONYM",
                max_height=args.max_height
            )
