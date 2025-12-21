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


def generate_heatmap(model_name: str, output_dir: str, result_dir: str) -> None:
    """
    Generate a heatmap for a given model showing accuracy across translate and noise modes.

    Args:
        model_name: The model directory name (e.g., "gpt-5", "gpt-5-mini", "gpt-5-nano")
        output_dir: Directory to save the heatmap image (default: current directory)
        result_dir: Directory containing the score files (default: "result/score")
    """

    # Initialize data structure: dict[translate_mode][noise_mode] = accuracy
    data_dict = {}
    for tm in translate_modes:
        data_dict[tm] = {}
        for nm in noise_modes:
            data_dict[tm][nm] = None

    # Path to model's statistics directory
    model_score_dir = Path(result_dir) / model_name / "statistics"

    if not model_score_dir.exists():
        print(f"Error: Model directory '{model_score_dir}' does not exist")
        return

    # The file naming convention is now in JSON format:
    # ["BFCL_v4_multiple", language, translate_level, pre_translate, noise, prompt_translate, post_translate].json
    # Example: ["BFCL_v4_multiple","zh","fulltrans","nopretrans","nonoise","noprompt","noposttrans"].json

    for score_file in model_score_dir.glob("*.json"):
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                accuracy = data.get("accuracy")

                if accuracy is None:
                    print(f"Warning: No 'accuracy' field in {score_file.name}")
                    continue

                # Extract translate and noise modes from filename
                filename = score_file.stem  # Remove .jsonl extension

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

                # Map combination of tags to translate_mode
                # NT = en + na
                if language_tag == "en" and translate_level_tag == "na":
                    translate_mode = "NT"
                # PAR = (zh or hi or igbo) + parttrans
                elif language_tag in ["zh", "hi", "igbo"] and translate_level_tag == "parttrans":
                    translate_mode = "PAR"
                # All other cases require fulltrans
                elif language_tag in ["zh", "hi", "igbo"] and translate_level_tag == "fulltrans":
                    # FT = fulltrans + nopretrans + noprompt + noposttrans
                    if (pre_translate_tag == "nopretrans" and prompt_translate_tag == "noprompt" and
                        post_translate_tag == "noposttrans"):
                        translate_mode = "FT"
                    # PT = fulltrans + nopretrans + prompt + noposttrans
                    elif (pre_translate_tag == "nopretrans" and prompt_translate_tag == "prompt" and
                          post_translate_tag == "noposttrans"):
                        translate_mode = "PT"
                    # PRE = fulltrans + pretrans + noprompt + noposttrans
                    elif (pre_translate_tag == "pretrans" and prompt_translate_tag == "noprompt" and
                          post_translate_tag == "noposttrans"):
                        translate_mode = "PRE"
                    # POST = fulltrans + nopretrans + noprompt + posttrans
                    elif (pre_translate_tag == "nopretrans" and prompt_translate_tag == "noprompt" and
                          post_translate_tag == "posttrans"):
                        translate_mode = "POST"
                    else:
                        print(f"Warning: Unknown tag combination in {score_file.name}")
                        print(f"  Tags: pre={pre_translate_tag}, prompt={prompt_translate_tag}, post={post_translate_tag}")
                        continue
                else:
                    print(f"Error: Unknown language/translate_level combination in {score_file.name}")
                    print(f"  language={language_tag}, translate_level={translate_level_tag}")
                    exit(1)

                # Store the accuracy
                data_dict[translate_mode][noise_mode] = accuracy
                print(f"Loaded {score_file.name}: {translate_mode} + {noise_mode} = {accuracy:.3f}")

        except Exception as e:
            print(f"Error reading {score_file.name}: {e}")

    # Convert to DataFrame
    data = []
    for tm in translate_modes:
        row = []
        for nm in noise_modes:
            value = data_dict[tm][nm]
            row.append(value if value is not None else np.nan)
        data.append(row)

    df = pd.DataFrame(data, index=translate_modes, columns=noise_modes)

    # Transpose the dataframe for heatmap visualization
    df = df.T

    # Check if we have any data
    if df.isna().all().all():
        print(f"Error: No valid data found for model '{model_name}'")
        return

    # Print summary
    print(f"\nData for model '{model_name}':")
    print(df)

    # Plot heatmap
    plt.figure(figsize=(8, 5))

    # Use a lighter, pleasant colormap
    plt.imshow(df, cmap="RdYlGn", interpolation="nearest", vmin=0.0, vmax=1.0)

    # Colorbar
    plt.colorbar(label="Accuracy")

    # Ticks (transposed: translate modes on x-axis, noise modes on y-axis)
    plt.xticks(np.arange(len(translate_modes)), translate_modes, rotation=45)
    plt.yticks(np.arange(len(noise_modes)), noise_modes)

    # Annotate values in each grid cell
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iloc[i, j]
            # Only annotate if we have data
            if not pd.isna(value):
                plt.text(
                    j, i,
                    f"{value:.3f}",             # round to 3 decimals
                    ha="center", va="center",
                    color="black", fontsize=9   # black text = readable on light colormap
                )

    plt.title(f"Heatmap: {model_name} - Translate Mode × Noise Mode")
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"heatmap_{model_name}.png")
    plt.savefig(output_path)
    print(f"\nSaved heatmap to {output_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate heatmaps showing accuracy across translate and noise modes for specified models."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Model names to process (e.g., gpt-5 gpt-5-mini gpt-5-nano)"
    )
    parser.add_argument(
        "--output-dir",
        default="tool/plots/heatmaps",
        help="Directory to save heatmap images (default: current directory)"
    )
    parser.add_argument(
        "--result-dir",
        default="tool/result",
        help="Directory containing the result files (default: tool/result)"
    )

    args = parser.parse_args()

    for model in args.models:
        print(f"\n{'='*60}")
        print(f"Generating heatmap for {model}")
        print(f"{'='*60}")
        generate_heatmap(model, args.output_dir, args.result_dir)