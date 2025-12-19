"""
Generate CSV file for validating fully translated dataset.
Creates a table with columns: Case ID, Original, Translated, Same Meaning? (Y/N), Correct Translation
"""

import argparse
import csv
from src_py.utils import load_json_lines_from_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV for validating fully translated dataset"
    )
    parser.add_argument(
        "--original",
        default="tool/dataset/BFCL_v4_multiple.jsonl",
        help="Original dataset file (default: tool/dataset/BFCL_v4_multiple.jsonl)"
    )
    parser.add_argument(
        "--translated",
        default="tool/dataset/BFCL_v4_multiple_igbo_full_gpt5.json",
        help="Fully translated dataset file (default: tool/dataset/BFCL_v4_multiple_igbo_full_gpt5.json)"
    )
    parser.add_argument(
        "--output",
        default="validation_full.csv",
        help="Output CSV file (default: validation_full.csv)"
    )

    args = parser.parse_args()

    # Load datasets
    print(f"Loading original dataset from {args.original}...")
    original_data = load_json_lines_from_file(args.original)
    original_map = {item["id"]: item for item in original_data}

    print(f"Loading translated dataset from {args.translated}...")
    translated_data = load_json_lines_from_file(args.translated)

    # Create CSV
    print(f"Generating CSV file {args.output}...")
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow([
            "Case ID",
            "Original",
            "Translated",
            "Same Meaning? (Y/N)",
            "Correct Translation (Copy from Translated if no modification needed)"
        ])

        # Write data rows
        for translated_item in translated_data:
            case_id = translated_item["id"]

            # Get original question
            if case_id not in original_map:
                print(f"Warning: {case_id} not found in original dataset, skipping...")
                continue

            original_question = original_map[case_id]["question"][0][0]["content"]
            translated_question = translated_item["question"][0][0]["content"]

            writer.writerow([
                case_id,
                original_question,
                translated_question,
                "",  # Empty for human to fill
                ""   # Empty for human to fill
            ])

    print(f"✅ CSV file generated: {args.output}")
    print(f"Total rows: {len(translated_data)}")


if __name__ == "__main__":
    main()
