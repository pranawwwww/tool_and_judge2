
import json
import os
import uuid
from codebase_rs import *
os.environ['HF_HOME'] = "/work/nvme/bfdz/zluo8/huggingface"
from dotenv import load_dotenv
from src_py.utils import load_config_from_file
from src_py.utils import load_json_lines_from_file
from src_py.utils import combine_entries_to_pairs
from src_py.utils import get_model_directory_safe_name
from src_py.vllm_backend import create_vllm_backend
from codebase_rs import concatenate_preference_datasets, concatenate_perplexity_datasets, dispatch_preference_results, dispatch_perplexity_results
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
args = parser.parse_args()

# Load config from specified file
if not args.config:
    print("Error: Please specify a config file using --config argument. For example, --config config1.py")
    exit(1)

# Run maturin develop to build and install the Rust extension
print("Building Rust extension with maturin develop...")
result = subprocess.run(["maturin", "develop"], check=True)

# Now import and use the module
time.sleep(2)  # Give some time for the build to complete


print(f"Loading config from: {args.config}")
config: JudgeConfig = load_config_from_file(args.config, "config")
print("Processing configuration: ", config)


# Get or create backend with caching
model_name = config.model.to_string() # Get model name from Rust enum
model_safe_name = get_model_directory_safe_name(model_name)
# Start the first pass

experiment_str = config.experiment.to_string()
combined_input_path = f"judge/result/{model_safe_name}/{experiment_str}_input.jsonl"
combined_output_path = f"judge/result/{model_safe_name}/{experiment_str}_output.jsonl"

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
        concatenate_preference_datasets(model_safe_name, first_lang, second_lang, combined_input_path)
        combined_entries = load_json_lines_from_file(combined_input_path)
        if len(combined_entries) == 0:
            print(f"All entries for experiment {experiment_str} have been processed. Exiting.")
            exit(0)
        # preference uses vllm backend
        engine, tokenizer = create_vllm_backend(model_name, args.num_gpus)
        semaphore = asyncio.Semaphore(200)
        async def collect_single_preference_async(entry: dict) -> dict:
            """
            entry is of type TwoAnswersEntry in src/judge/generate_dataset.rs 
            """
            async with semaphore:
                if config.model == LocalModel.Llama3_3_70B:
                    from src_py.llama3_1_backend import collect_preference_local_async
                # elif config.model == LocalModel.Qwen3_14B:
                #     from src_py.qwen3_backend import collect_preference_local_async
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
            pending_results = []
            completed_count = 0
            with open(combined_output_path, 'w', encoding='utf-8') as f:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    pending_results.append(result)
                    completed_count += 1
                    if len(pending_results) >= 10:
                        for r in pending_results:
                            f.write(json.dumps(r, ensure_ascii=False) + '\n')
                            f.flush()
                        print(f"Written {completed_count}/{len(combined_entries)} entries to file")
                        pending_results = []
                # Write any remaining results
                if pending_results:
                    for r in pending_results:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                        f.flush()
                    print(f"Written {completed_count}/{len(combined_entries)} entries to file")
        asyncio.run(collect_all_preference_entries())
        # dispatch results
        dispatch_preference_results(model_safe_name, config.lang1, config.lang2, combined_output_path)
        # delete this file
        os.remove(combined_output_path)
        print(f"Dispatched results and removed file: {combined_output_path}")     
    case JudgeExperiment.Perplexity(lang=lang):
        if os.path.exists(combined_output_path):
            dispatch_perplexity_results(model_safe_name, lang, combined_output_path)
            # delete this file
            os.remove(combined_output_path)
            print(f"Dispatched results from existing file: {combined_output_path}")

        # call rust function to concatenate two datasets
        concatenate_perplexity_datasets(model_safe_name, lang, combined_input_path)
        combined_entries = load_json_lines_from_file(combined_input_path)
        if len(combined_entries) == 0:
            print(f"All entries for experiment {experiment_str} have been processed. Exiting.")
            exit(0)
        # perplexity uses huggingface backend
        model_size = config.model.size_in_billion_parameters()
        batch_size = 120 * args.num_gpus / model_size  # model_size * batch size = 120 * num_gpus
        model, tokenizer = create_huggingface_backend(model_name, batch_size)
        # combined_entries has type TwoAnswersEntry in src/judge/generate_dataset.rs
        # output entries have type PerplexityResultEntry in src/judge/result_file_model.rs
        # TODO








