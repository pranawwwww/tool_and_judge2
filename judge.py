
import json
import os
import uuid

os.environ['HF_HOME'] = "/work/nvme/bfdz/zluo8/huggingface"
from dotenv import load_dotenv
from src_py.utils import load_config_from_file
from src_py.utils import load_json_lines_from_file
from src_py.utils import combine_entries_to_pairs
from src_py.utils import get_model_directory_safe_name
from src_py.vllm_backend import create_vllm_backend
from src_py.utils import calculate_perplexity_from_logits

import argparse
import subprocess
import time
import asyncio
load_dotenv(".env")




# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run BFCL evaluation with custom configuration"
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to a Python file containing the 'config'"
)
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for local inference (default: 1)"
)
parser.add_argument(
    "--debug-limit",
    type=int,
    default=None,
    help="Limit the number of entries to process for debugging (default: None)"
)

args = parser.parse_args()

# Load config from specified file
if not args.config:
    print("Error: Please specify a config file using --config argument. For example, --config config1.py")
    exit(1)

# Run maturin develop to build and install the Rust extension with file locking
import fcntl
lock_file_path = "/tmp/maturin_build_lock"
print("Acquiring build lock...")
with open(lock_file_path, "w") as lock_file:
    # Acquire exclusive lock (blocks until available)
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    try:
        print("Building Rust extension with maturin develop...")
        # result = subprocess.run(["maturin", "develop"], check=True)
        result = subprocess.run(["maturin", "develop", "--release"], check=True)
        print("Installed Rust extension successfully.")
        time.sleep(2)  # Give some time for the build to complete
    finally:
        # Release lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        print("Released build lock.")

# from codebase_rs import concatenate_preference_datasets, concatenate_perplexity_datasets, dispatch_preference_results, dispatch_perplexity_results
from codebase_rs import *

print(f"Loading config from: {args.config}")
config: JudgeConfig = load_config_from_file(args.config, "config")
print("Processing configuration: ", config)


model_name = config.model.to_string() # Get model name from Rust enum
model_safe_name = get_model_directory_safe_name(model_name)
# Start the first pass

experiment_str = config.experiment.to_string()
combined_input_path = f"judge/result/{model_safe_name}/{experiment_str}_input.jsonl"
combined_output_path = f"judge/result/{model_safe_name}/{experiment_str}_output.jsonl"

main_vllm_backend_created = False
main_hf_backend_created = False
assistant_api_backend_created = False
main_vllm_engine = None
main_hf_model = None
main_tokenizer = None
assistant_client = None



async def main_async():
    match config.experiment:
        case JudgeExperiment.PreferenceDirect(lang1=lang1, lang2=lang2):
            # Determine alphabetical order for language codes
            sorted_langs = sorted([lang1, lang2])
            first_lang = sorted_langs[0]
            second_lang = sorted_langs[1]
            # load two answers datasets

            # generate a filename based on uuid
            # uuid_str = str(uuid.uuid4())
            
            # check if there is file at combined_output_path
            if os.path.exists(combined_output_path):
                dispatch_preference_results(model_safe_name, lang1, lang2, combined_output_path)
                # delete this file
                os.remove(combined_output_path)
                print(f"Dispatched results from existing file: {combined_output_path}")
            
            # call rust function to concatenate two datasets
            concatenate_preference_datasets(model_safe_name, first_lang, second_lang, combined_input_path, debug_limit=args.debug_limit)
            combined_entries = load_json_lines_from_file(combined_input_path)
            semaphore = asyncio.Semaphore(200)
            async def collect_single_preference_async(entry: dict) -> dict:
                """
                entry is of type TwoAnswersEntry in src/judge/generate_dataset.rs 
                """
                global main_vllm_backend_created, main_vllm_engine, main_tokenizer
                if not main_vllm_backend_created:
                    print(f"Creating VLLM backend for model {model_name} using {args.num_gpus} GPUs...", flush=True)
                    main_vllm_engine, main_tokenizer = create_vllm_backend(model_name, args.num_gpus)
                    print(f"VLLM backend created for model {model_name}", flush=True)
                    main_vllm_backend_created = True
                engine = main_vllm_engine
                tokenizer = main_tokenizer
                async with semaphore:
                    if config.model == LocalModel.Llama3_3_70B:
                        from src_py.llama3_1_backend import collect_preference_local_async
                    elif config.model in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b]:
                        from src_py.qwen3_backend import collect_preference_local_async
                    else:
                        raise ValueError(f"Unsupported model for preference collection: {config.model}")
                    try:
                        logprob1, logprob2 = await collect_preference_local_async(entry['question'], entry['answer1'], entry['answer2'], engine, tokenizer)
                        if logprob1 >= logprob2:
                            preferred_answer = 1
                        else:
                            preferred_answer = 2
                        logprob_signed_difference = logprob1 - logprob2
                        preference = {
                            'Ok': {
                                'preferred_answer': preferred_answer,
                                'logprob_signed_difference': logprob_signed_difference,
                                'logprob1': logprob1,
                                'logprob2': logprob2,
                            }
                        }
                    except Exception as e:
                        error_message = str(e)
                        preference = {
                            'Err': error_message
                        }
                    # The output type is PreferenceResultEntry in src/judge/result_file_model.rs
                    return {
                        "index": entry["index"],
                        "preference": preference,
                        "question": entry["question"],
                        "answer1": entry["answer1"],
                        "answer2": entry["answer2"],
                        "lang1": entry["lang1"],
                        "lang2": entry["lang2"],
                        "is_correct1": entry["is_correct1"],
                        "is_correct2": entry["is_correct2"],
                        "subject": entry["subject"],
                    }
            
            async def collect_all_preference_entries() -> list[dict]:
                tasks = [collect_single_preference_async(entry) for entry in combined_entries]
                with open(combined_output_path, 'w', encoding='utf-8') as f:
                    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                        result = await coro
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
                        print(f"Written {i}/{len(combined_entries)} entries to file")
            asyncio.run(collect_all_preference_entries())

            # dispatch results
            dispatch_preference_results(model_safe_name, lang1, lang2, combined_output_path)
            # delete this file
            os.remove(combined_output_path)
            print(f"Dispatched results and removed file: {combined_output_path}")
        case JudgeExperiment.Perplexity(lang=lang):
            # collect the response (to the response folder)
            # we only need the questions for the first pass, but will need the two answers for the second pass
            # however, we need two answers of the same language, so we can only load two one-answer datasets and combine them
            # we do not concatenate the datasets because two conjugated cases should be processed at the same time.

            # Use the existing filtering before letting LLMs to merge the dataset
            # Assume all filtered entries can be successfully processed


            # this pass does not need to check existing results from aggregated file

            # generate the dataset if not exists
            generate_two_answers_same_lang_dataset(lang)
            # load the dataset
            dataset_entries = load_json_lines_from_file(f"judge/datasets/two_answers_same_lang/{lang}.jsonl")
            dataset_entries = {entry['index']: entry for entry in dataset_entries}
            # collect the existing results from the result file
            response_result_path = f"judge/result/{model_safe_name}/response/{lang}.jsonl"
            # create directory if not exists
            os.makedirs(os.path.dirname(response_result_path), exist_ok=True)

            try:
                existing_results = load_json_lines_from_file(response_result_path)
                existing_result_entries = {entry['index']: entry for entry in existing_results}
            except FileNotFoundError:
                existing_result_entries = {} 
            
            indices_to_process = []
            for index, entry in dataset_entries.items():
                if index not in existing_result_entries:
                    indices_to_process.append(index)
            print(f"Total entries in dataset for language {lang}: {len(dataset_entries)}")
            print(f"Existing results in response file: {len(existing_result_entries)}")
            print(f"Entries to process in this run: {len(indices_to_process)}")
            if args.debug_limit is not None:
                indices_to_process = indices_to_process[: args.debug_limit]
                print(f"Debug limit applied, processing only first {args.debug_limit} entries")
            
            # Process entries in batches
            model_size = config.model.size_in_billion_parameters()
            # Reduce batch size significantly for HuggingFace backend due to padding overhead
            # With max_length=2048 and padding, memory usage is batch_size * 2048 * hidden_size
            batch_size = max(1, int(60 * args.num_gpus / model_size))  # Reduced from 120 to 60
            print(f"Processing {len(indices_to_process)} entries...", flush=True)
            total_processed = 0
            with open(response_result_path, 'a', encoding='utf-8') as f:
                for i in range(0, len(indices_to_process), batch_size):
                    batch_indices = indices_to_process[i:i+batch_size]
                    batch_entries = [dataset_entries[index] for index in batch_indices]
                    print(f"Processing batch {i//batch_size + 1}/{(len(indices_to_process) + batch_size - 1)//batch_size}", flush=True)

                    # Get model outputs for the batch
                    if config.model == LocalModel.Llama3_3_70B:
                        from src_py.llama3_1_backend import collect_response_batch
                    elif config.model in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b]:
                        from src_py.qwen3_backend import collect_response_batch
                    else:
                        raise ValueError(f"Unsupported model for response collection: {config.model}")

                    batch_outputs = collect_response_batch(batch_entries, args.num_gpus, config.model)

                    # Process each entry in the batch and write immediately
                    for entry, output in zip(batch_entries, batch_outputs):
                        result_entry = {
                            'index': entry['index'],
                            'question': entry['question'],
                            'response': output,
                            'lang': entry['lang'],
                            'subject': entry['subject'],
                        }
                        # Write result immediately
                        f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                        f.flush()
                        total_processed += 1

                    # Flush after each batch to ensure results are written
                    f.flush()
                    print(f"Written {total_processed}/{len(indices_to_process)} entries to file", flush=True)
            print(f"Completed writing all {total_processed} responses to {response_result_path}")



            
            # call gpt-5/deepseek to merge the style with the ground truth (to the perplexity_dataset folder)

            response_entries = load_json_lines_from_file(response_result_path)
            response_entries = {entry['index']: entry for entry in response_entries}

            perplexity_dataset_path = f"judge/result/{model_safe_name}/perplexity_dataset/{lang}.jsonl"

            # create directory if not exists
            os.makedirs(os.path.dirname(perplexity_dataset_path), exist_ok=True)
            # first get the existing file content
            # TODO

            # collect the perplexity from the processed entries (to the perplexity folder)
            if os.path.exists(combined_output_path):
                dispatch_perplexity_results(model_safe_name, lang, combined_output_path)
                # delete this file
                os.remove(combined_output_path)
                print(f"Dispatched results from existing file: {combined_output_path}")

            # call rust function to concatenate two datasets
            # TODO: This should be replaced by loading from perplexity_dataset folder
            concatenate_perplexity_datasets(model_safe_name, lang, combined_input_path, debug_limit=args.debug_limit)
            combined_entries = load_json_lines_from_file(combined_input_path)
            if len(combined_entries) == 0:
                print(f"All entries for experiment {experiment_str} have been processed. Exiting.")
                exit(0)
            # perplexity uses huggingface backend
            from src_py.huggingface_backend import create_huggingface_backend
            
            print(f"Creating backend for model {model_name} with batch size {batch_size}...", flush=True)
            model, tokenizer = create_huggingface_backend(model_name, batch_size)
            print(f"Backend created for model {model_name} with batch size {batch_size}", flush=True)
            # combined_entries has type OneAnswerEntry in src/judge/generate_dataset.rs
            # output entries have type PerplexityResultEntry in src/judge/result_file_model.rs

            

            # Process entries in batches
            print(f"Processing {len(combined_entries)} entries with batch size {batch_size}", flush=True)
            total_processed = 0

            # Open file for writing results incrementally
            with open(combined_output_path, 'w', encoding='utf-8') as f:
                for i in range(0, len(combined_entries), batch_size):
                    batch_entries = combined_entries[i:i+batch_size]
                    print(f"Processing batch {i//batch_size + 1}/{(len(combined_entries) + batch_size - 1)//batch_size}", flush=True)

                    # Get model outputs for the batch
                    if config.model == LocalModel.Llama3_3_70B:
                        from src_py.llama3_1_backend import collect_perplexity_batch
                    elif config.model in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b]:
                        from src_py.qwen3_backend import collect_perplexity_batch
                    else:
                        raise ValueError(f"Unsupported model for perplexity collection: {config.model}")

                    batch_outputs = collect_perplexity_batch(batch_entries, model, tokenizer)

                    # Process each entry in the batch and write immediately
                    for entry, output in zip(batch_entries, batch_outputs):
                        try:
                            perplexity = calculate_perplexity_from_logits(
                                output['logits'],
                                output['input_ids'],
                                output['answer'],
                                tokenizer
                            )

                            result_entry = {
                                'index': entry['index'],
                                'perplexity': {'Ok': perplexity},
                                'question': entry['question'],
                                'answer': entry['answer'],
                                'lang': entry['lang'],
                                'is_correct': entry['is_correct'],
                                'subject': entry['subject'],
                            }
                        except Exception as e:
                            error_message = str(e)
                            result_entry = {
                                'index': entry['index'],
                                'perplexity': {'Err': error_message},
                                'question': entry['question'],
                                'answer': entry['answer'],
                                'lang': entry['lang'],
                                'is_correct': entry['is_correct'],
                                'subject': entry['subject'],
                            }

                        # Write result immediately
                        f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                        total_processed += 1

                    # Flush after each batch to ensure results are written
                    f.flush()
                    print(f"Written {total_processed}/{len(combined_entries)} entries to file", flush=True)

            print(f"Completed writing all {total_processed} results to {combined_output_path}")

            # Dispatch results
            dispatch_perplexity_results(model_safe_name, lang, combined_output_path)
            # Delete this file
            os.remove(combined_output_path)
            print(f"Dispatched results and removed file: {combined_output_path}")

if __name__ == "__main__":
    asyncio.run(main_async())






