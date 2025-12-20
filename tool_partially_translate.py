"""
Async version of partial translation with support for DeepSeek and GPT-5.
Supports multiple languages: Igbo, Chinese (Simplified), Hindi.
Processes multiple translations concurrently for better performance.
"""

import argparse
import asyncio
import json
import os
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from src_py.utils import load_json_lines_from_file

# Load API keys from .env file
load_dotenv(dotenv_path=".env")


def create_translation_prompt(
    question_content: str,
    keywords: list[str],
    target_language: str
) -> str:
    """
    Create a prompt for translation (works for both DeepSeek and GPT-5).
    The model will output only the final translation.
    """
    keywords_str = json.dumps(keywords, ensure_ascii=False)

    prompt = f"""Translate the following English question to {target_language}. Keep the keywords listed below unchanged (do not translate them).

Question:
{question_content}

Keywords to preserve (keep in English):
{keywords_str}

Provide only the {target_language} translation."""

    return prompt


async def translate_with_api(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    api_type: str
) -> str | None:
    """
    Use API to translate the question (supports DeepSeek and GPT-5).

    Args:
        client: AsyncOpenAI client
        prompt: The translation prompt
        model: Model name to use
        api_type: "deepseek" or "gpt5"

    Returns:
        Translated text or None if translation fails
    """
    try:
        if api_type == "gpt5":
            # GPT-5 uses responses API
            response = await client.responses.create(
                input=[
                    {"role": "developer", "content": "You are a professional translator specializing in partial translations for technical function calling tasks."},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                store=False
            )
            translated_text = response.output_text.strip() if hasattr(response, 'output_text') else None
        else:
            # DeepSeek uses chat completions API
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            message = response.choices[0].message
            translated_text = message.content.strip() if message.content else None

        return translated_text

    except Exception as e:
        print(f"Error calling API: {e}")
        return None


async def translate_item(
    client: AsyncOpenAI,
    dataset_line: dict,
    keywords_map: dict,
    model: str,
    api_type: str,
    target_language: str,
    index: int,
    total: int,
    semaphore: asyncio.Semaphore
) -> tuple[str, dict | None]:
    """
    Translate a single item asynchronously with semaphore control.

    Returns:
        Tuple of (item_id, translated_line or None)
    """
    async with semaphore:
        item_id = dataset_line["id"]
        question_content = dataset_line["question"][0][0]["content"]

        # Get keywords
        if item_id not in keywords_map:
            print(f"Warning: No keywords found for {item_id}, skipping...")
            return (item_id, None)

        keywords = keywords_map[item_id]

        # Create translation prompt
        prompt = create_translation_prompt(question_content, keywords, target_language)

        # Translate
        print(f"[{index+1}/{total}] Translating {item_id} to {target_language}...")
        translated_content = await translate_with_api(client, prompt, model, api_type)

        if translated_content:
            # Create translated item
            translated_line = dataset_line.copy()
            translated_line["question"] = [[{
                "role": "user",
                "content": translated_content
            }]]

            print(f"  ✓ {item_id}: {translated_content[:60]}...")
            return (item_id, translated_line)
        else:
            print(f"  ✗ {item_id}: Translation failed")
            return (item_id, None)




async def main():
    # === Parse command line arguments ===
    parser = argparse.ArgumentParser(
        description="Async partial translation with DeepSeek or GPT-5"
    )
    parser.add_argument(
        "--language",
        choices=["igbo", "chinese", "hindi"],
        default="igbo",
        help="Target language (igbo, chinese, or hindi)"
    )
    parser.add_argument(
        "--api",
        choices=["deepseek", "gpt5"],
        default="gpt5",
        help="API to use (deepseek or gpt5)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: deepseek-chat for deepseek, gpt-5-nano for gpt5)"
    )
    parser.add_argument(
        "--input",
        default="tool/dataset/BFCL_v4_multiple.jsonl",
        help="Input dataset file"
    )
    parser.add_argument(
        "--keywords",
        default="tool/dataset/BFCL_v4_multiple_keywords.jsonl",
        help="Keywords file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (default: auto-generated)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=200,
        help="Maximum number of concurrent translations (default: 200)"
    )

    args = parser.parse_args()

    # === Configuration ===
    api_type = args.api
    language_map = {
        "igbo": "Igbo",
        "chinese": "Chinese (Simplified)",
        "hindi": "Hindi"
    }
    target_language = language_map[args.language]

    # Set default model based on API
    if args.model:
        model_name = args.model
    else:
        model_name = "deepseek-chat" if api_type == "deepseek" else "gpt-5-nano"

    # Set default output file
    if args.output:
        output_file = args.output
    else:
        output_file = f"tool/dataset/BFCL_v4_multiple_{args.language}_partial_{api_type}.json"

    # Get API configuration
    if api_type == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com"
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY not found in .env")
    else:  # gpt5
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in .env")

    print(f"Using {api_type.upper()} API with model: {model_name}")
    print(f"Max concurrent translations: {args.max_concurrent}")

    # Initialize async client
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # === Load input files ===
    print(f"\nLoading dataset from {args.input}...")
    dataset = load_json_lines_from_file(args.input)

    print(f"Loading keywords from {args.keywords}...")
    keywords_data = load_json_lines_from_file(args.keywords)
    # Create a mapping of id to keywords
    keywords_map = {kw['id']: kw['keywords'] for kw in keywords_data}

    # === Load existing translations (for resumption) ===
    existing_indices = set()
    try:
        existing_lines = load_json_lines_from_file(output_file)
        existing_indices = {item["id"] for item in existing_lines}
        print(f"Found {len(existing_indices)} existing translations, will skip those")
    except FileNotFoundError:
        print(f"No existing translations found, starting fresh")
        # Create empty file
        with open(output_file, "w", encoding="utf-8") as f:
            pass

    # Filter out already processed items
    items_to_process = [
        (i, item) for i, item in enumerate(dataset)
        if item["id"] not in existing_indices
    ]

    if not items_to_process:
        print("\n✅ All items already translated!")
    else:
        print(f"\nProcessing {len(items_to_process)} new items...")

        # === Process with semaphore and asyncio.as_completed ===
        total = len(dataset)
        semaphore = asyncio.Semaphore(args.max_concurrent)
        failed = []
        completed = 0

        # Create all tasks
        tasks = [
            translate_item(client, item, keywords_map, model_name, api_type, target_language, idx, total, semaphore)
            for idx, item in items_to_process
        ]

        # Process tasks as they complete
        for coro in asyncio.as_completed(tasks):
            item_id, translated_line = await coro
            completed += 1

            if translated_line:
                # Append immediately to file
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(translated_line, ensure_ascii=False) + "\n")
            else:
                failed.append(item_id)

            if completed % 10 == 0 or completed == len(items_to_process):
                print(f"Progress: {completed}/{len(items_to_process)} completed")

        # Summary
        if failed:
            print(f"\n⚠ Warning: {len(failed)} translations failed:")
            for item_id in failed[:10]:
                print(f"  - {item_id}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")

        print(f"\n✅ Translation complete!")
        print(f"Success rate: {completed - len(failed)}/{completed}")

    # Sort the output file by ID (always runs)
    print(f"\nSorting output file by ID...")
    all_lines = load_json_lines_from_file(output_file)
    sorted_lines = sorted(
        all_lines,
        key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf')
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for line in sorted_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
