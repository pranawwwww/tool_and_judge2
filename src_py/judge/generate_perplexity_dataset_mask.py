import os
import json
import asyncio
import dotenv

def generate_perplexity_dataset_mask():
    dotenv.load_dotenv(".env")
    output_path = f'judge/datasets/perplexity_mask.jsonl'
    input_dataset_path = f'judge/datasets/mmmlu_normalized/en.jsonl' # use English as the benchmark

    # Load existing results if the file exists
    existing_results = {}
    if os.path.exists(output_path):
        print(f"Loading existing results from {output_path}...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                existing_results[entry['original_index']] = entry
        print(f"Loaded {len(existing_results)} existing results.")

    # read all lines and json parse them
    with open(input_dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    dataset_entries = [json.loads(line) for line in lines]

    # Filter out entries that already have results
    entries_to_process = [entry for entry in dataset_entries if entry['original_index'] not in existing_results]
    print(f"Total entries: {len(dataset_entries)}, Already processed: {len(existing_results)}, To process: {len(entries_to_process)}")

    # If all entries are already processed, just sort and return
    if not entries_to_process:
        print("All entries already processed. Sorting and saving final results...")
        sorted_results = sorted(existing_results.values(), key=lambda x: x['original_index'])
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in sorted_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Sorted results saved to {output_path}")
        return

    # Create gpt5 client connection
    from openai import AsyncOpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY not found in environment variables"
    print("Creating GPT-5 client...")
    client = AsyncOpenAI(
        api_key=api_key,
    )
    print("GPT-5 client created.")

    # Create a semaphore to limit concurrent tasks
    MAX_CONCURRENT_TASKS = 200
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    # create async runtime
    async def judge_qa_pair_async(entry: dict) -> dict:
        async with semaphore:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "The user will provide a multiple choice question, 4 answer choices and the correct answer. "
                        "Your task is to judge whether there is a reasonably high chance for a sufficiently intelligent being "
                        "to output the exact correct answer among the four choices, even if the choices are not given."
                        "Questions that fail to meet this requirement may include: \n"
                        "- Questions that contain 'Which of the following...' or similar phrasing. This makes it impossible to hit the correct answer without seeing the choices. For example, 'Which of the following has a red color?' A: apple. B:...\n"
                        "- Questions that are open-ended to an extent that the correct answer cannot be hit with a reasonable chance without seeing the choices. For example, Shakespeare has the motto ___. A: To thine own self be true. B:...\n"
                        "Please note that if a question has a fixed answer but with multiple possible phrasings, it is still valid. For example, What was the most important finding by the House of Lords in the Pinochet case? A: The Pinochet case confirmed that all public acts enjoy immunity. B:...\n"
                        "Your output should only contain either 'VALID' or 'INVALID', without any additional explanation."
                    )
                },
                {
                    "role": "user",
                    "content": (f"Question: {entry['question']}\n"
                                f"A: {entry['choices'][0]}\n"
                                f"B: {entry['choices'][1]}\n"
                                f"C: {entry['choices'][2]}\n"
                                f"D: {entry['choices'][3]}\n"
                                f"Correct Answer: {entry['choices'][entry['answer']]}\n"
                                f"Does this question meet the requirement? Please only output 'VALID' or 'INVALID'.")
                }
            ]
            response = await client.chat.completions.create(
                model="gpt-5",
                messages=messages,
            )
            response_str = response.choices[0].message.content
            if response_str == "VALID":
                is_valid = True
            elif response_str == "INVALID":
                is_valid = False
            else:
                print(f"Unexpected response: {response_str}. Marking as INVALID.")
                is_valid = False
            print(f"Question {entry['original_index']}: {entry['question'][:50]}... judged as {'VALID' if is_valid else 'INVALID'}")

            result = {
                "index": entry["original_index"],
                "valid": is_valid,
                "question": entry["question"],
                "choices": entry["choices"],
                "subject": entry["subject"],
            }

            return result

    # Run the async function for all entries and write in batches
    async def process_all_entries():
        tasks = [judge_qa_pair_async(entry) for entry in entries_to_process]
        pending_results = []
        completed_count = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            pending_results.append(result)
            completed_count += 1

            # Write to file every 10 entries
            if len(pending_results) >= 10:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for r in pending_results:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                print(f"Written {completed_count}/{len(entries_to_process)} entries to file")
                pending_results = []

        # Write any remaining results
        if pending_results:
            with open(output_path, 'a', encoding='utf-8') as f:
                for r in pending_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"Written {completed_count}/{len(entries_to_process)} entries to file (final batch)")

    # start the runtime
    print(f"Processing {len(entries_to_process)} entries...")
    asyncio.run(process_all_entries())

    # Load all results (existing + new) and sort by index
    print("Processing complete. Sorting all results...")
    all_results = {}
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            all_results[entry['index']] = entry

    # Sort by index and write back to file
    sorted_results = sorted(all_results.values(), key=lambda x: x['index'])
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in sorted_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Generated and sorted perplexity dataset mask at {output_path} ({len(sorted_results)} total entries)")

if __name__ == "__main__":
    generate_perplexity_dataset_mask()