import argparse
import asyncio
import subprocess
import time

from src_py.utils import *

from dotenv import load_dotenv
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

# Check that config file is specified before building
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
        result = subprocess.run(["maturin", "develop", "--release"], check=True)
        print("Installed Rust extension successfully.")
        # time.sleep(5)  # Give some time for the build to complete
    finally:
        # Release lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        print("Released build lock.")

from codebase_rs import *
from src_py.api_backend import create_api_backend
from src_py.vllm_backend import create_vllm_backend

# Load config from specified file AFTER building Rust extension
# This is important because the config file imports codebase_rs
print(f"Loading config from: {args.config}")
config = load_config_from_file(args.config, "config")
print("Processing configuration: ", config)

# There is no elegant way in Python to construct and use a union type (no compile time type safety guaranteed)
# So we just nominally return every possible object
# This is the most robust way I can think of to avoid accessing the wrong variable
def create_backend(model: Model):
    client, engine, tokenizer, is_api = None, None, None, None
    match model:
        case Model.Api(api_model):
            print(f"Creating API backend for model {api_model}...")
            client = create_api_backend(api_model)
            print(f"API backend created successfully for model {api_model}")
            is_api = True
        case Model.Local(local_model):
            print(f"Creating vLLM backend for model {local_model}...")
            engine, tokenizer = create_vllm_backend(local_model, num_gpus=args.num_gpus)
            print(f"vLLM backend created successfully for model {local_model}")
            is_api = False
    return client, engine, tokenizer, is_api

# Global backend variables that persist across passes
main_backend_created = False
main_client = None
main_engine = None
main_tokenizer = None
main_is_api = None
assistant_backend_created = False
assistant_client = None
# Acquire model-level lock, so that all aggregated files can be free of race condition, and is safe to read

# Wrap everything in an async function because vllm can only run in the same async runtime
async def main_async():
    # Declare global variables that will be modified across passes
    global main_backend_created, main_client, main_engine, main_tokenizer, main_is_api
    global assistant_backend_created, assistant_client

    # The first pass is pre-translation
    # aggregate all questions across datasets
    # We can create a dataset file that contains questions only for each dataset file
    # aggregate and extract questions can be done at the same time
    # This involves calling a rust function
    # Then we have the question only dataset file. Its path can be retrieved from Rust code.
    # Then we get the python array object from reading the file
    print("----------PASS 1: PRE-TRANSLATE QUESTIONS----------")
    pass_pre_translate_prepare_aggregated_questions(config)
    aggregated_questions_input_file_path = pass_pre_translate_aggregated_questions_input_file_path(config)
    aggregated_questions_output_file_path = pass_pre_translate_aggregated_questions_output_file_path(config)
    if os.path.exists(aggregated_questions_output_file_path):
        pass_pre_translate_dispatch_results(config)
    # Each entry has a signature of type PreTranslateAggregatedInputQuestionEntry in src/tool/passes/pass_pre_translate.rs
    question_entries = load_json_lines_from_file(aggregated_questions_input_file_path)
    # Define all async functions first
    async def collect_single_question_translation_async(entry: dict, semaphore) -> dict:
        # Create backend on demand
        global main_backend_created, main_client, main_engine, main_tokenizer, main_is_api
        if not main_backend_created:
            main_client, main_engine, main_tokenizer, main_is_api = create_backend(config.model)
            main_backend_created = True

        async with semaphore:
            try:
                if config.model in [Model.Api(ApiModel.Gpt5), Model.Api(ApiModel.Gpt5Mini), Model.Api(ApiModel.Gpt5Nano)]:
                    from src_py.gpt5_backend import translate_tool_question_async
                    translated_question = await translate_tool_question_async(
                        model_name = config.model.to_string(),
                        client = main_client,
                        question = entry["question"],
                    )
                elif config.model in [Model.Api(ApiModel.DeepSeek)]:
                    from src_py.deepseek_backend import translate_tool_question_async
                    translated_question = await translate_tool_question_async(
                        model_name = config.model.to_string(),
                        client = main_client,
                        question = entry["question"],
                    )
                elif config.model in [Model.Local(LocalModel.Qwen3_8B), Model.Local(LocalModel.Qwen3_14B), Model.Local(LocalModel.Qwen3_32B), Model.Local(LocalModel.Qwen3_30bA3b), Model.Local(LocalModel.Qwen3Next80bA3b)]:
                    from src_py.qwen3_backend import translate_tool_question_async
                    translated_question = await translate_tool_question_async(
                        engine = main_engine,
                        tokenizer = main_tokenizer,
                        question = entry["question"],
                    )
                elif config.model in [Model.Local(LocalModel.Llama3_1_8B), Model.Local(LocalModel.Llama3_1_70B)]:
                    from src_py.llama3_1_backend import translate_tool_question_async
                    translated_question = await translate_tool_question_async(
                        engine = main_engine,
                        tokenizer = main_tokenizer,
                        question = entry["question"],
                    )
                elif config.model in [Model.Local(LocalModel.Granite4_0HTiny), Model.Local(LocalModel.Granite4_0HSmall)]:
                    from src_py.granite4_backend import translate_tool_question_async
                    translated_question = await translate_tool_question_async(
                        engine = main_engine,
                        tokenizer = main_tokenizer,
                        question = entry["question"],
                    )
                else:
                    raise NotImplementedError(f"Translation not implemented for model {config.model}.")
            except Exception as e:
                print(f"Error translating question index {entry['id']}: {e}")
                exit(1) # Todo: handle error properly
            # Each output entry has a signature of type PreTranslateAggregatedOutputQuestionEntry in src/tool/passes/pass_pre_translate.rs
            return {
                "id": entry["id"],
                "original_question": entry["question"],
                "translated_question": translated_question,
                "file_name": entry["file_name"],
            }
    async def collect_all_question_translations_async(entries: list[dict]) -> list[dict]:
        semaphore = asyncio.Semaphore(200)
        tasks = [collect_single_question_translation_async(entry, semaphore) for entry in entries]
        with open(aggregated_questions_output_file_path, "w") as f:
            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                result = await coro
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"Translated {i}/{len(entries)} questions...")

    await collect_all_question_translations_async(question_entries)
    # dispatch results to respective dataset files
    pass_pre_translate_dispatch_results(config)
    # finished processing pre translation, deleting input file
    os.remove(aggregated_questions_input_file_path)
    

    # Pre-translate is done

    # Then we call a python interface adapter to get translated questions
    # Then we actually do not need to replicate the same dataset file, but simply override the original questions with translated ones
    # dispatch result to separate result files


    # The second pass is generate the raw function calls
    # This involves first calling the rust function to generate the tool definitions for each model
    # Then aggregate all tool definitions across datasets
    # Then we take in the model-specific tool definitions and call the python interface adapter to get the raw function calls
    # We needs to manually map the function calls with its ids to generate the raw output file in python
    # Then dispatch result to separate result files
    # We generate the function name map and store it in a file for later use

    print("----------PASS 2: GENERATE RAW FUNCTION CALLS----------")
    pass_generate_raw_prepare_aggregated_input(config)
    aggregated_input_file_path = pass_generate_raw_aggregated_input_file_path(config)
    aggregated_output_file_path = pass_generate_raw_aggregated_output_file_path(config)
    if os.path.exists(aggregated_output_file_path):
        pass_generate_raw_dispatch_results(config)
    # Each entry has a signature of type GenerateRawAggregatedInputEntry in src/tool/passes/pass_generate_raw.rs
    input_entries = load_json_lines_from_file(aggregated_input_file_path)

    async def collect_single_raw_function_call_async(entry: dict, semaphore) -> dict:
        # Create backend on demand
        global main_backend_created, main_client, main_engine, main_tokenizer, main_is_api
        if not main_backend_created:
            main_client, main_engine, main_tokenizer, main_is_api = create_backend(config.model)
            main_backend_created = True

        async with semaphore:
            try:
                if config.model in [Model.Api(ApiModel.Gpt5), Model.Api(ApiModel.Gpt5Mini), Model.Api(ApiModel.Gpt5Nano)]:
                    from src_py.gpt5_backend import generate_tool_call_async
                    raw_output = await generate_tool_call_async(
                        model_name = config.model.to_string(),
                        client = main_client,
                        question=entry["question"],
                        tools = entry["tools"],
                        prompt_passing_in_english = entry["prompt_passing_in_english"],
                    )
                elif config.model in [Model.Api(ApiModel.DeepSeek)]:
                    from src_py.deepseek_backend import generate_tool_call_async
                    raw_output = await generate_tool_call_async(
                        model_name = config.model.to_string(),
                        client = main_client,
                        question=entry["question"],
                        tools = entry["tools"],
                        prompt_passing_in_english = entry["prompt_passing_in_english"],
                )
                elif config.model in [Model.Local(LocalModel.Qwen3_8B), Model.Local(LocalModel.Qwen3_14B), Model.Local(LocalModel.Qwen3_32B), Model.Local(LocalModel.Qwen3_30bA3b), Model.Local(LocalModel.Qwen3Next80bA3b)]:
                    from src_py.qwen3_backend import generate_tool_call_async
                    raw_output = await generate_tool_call_async(
                        engine = main_engine,
                        tokenizer = main_tokenizer,
                        question=entry["question"],
                        tools = entry["tools"],
                        prompt_passing_in_english = entry["prompt_passing_in_english"],
                )
                elif config.model in [Model.Local(LocalModel.Llama3_1_8B), Model.Local(LocalModel.Llama3_1_70B)]:
                    from src_py.llama3_1_backend import generate_tool_call_async
                    raw_output = await generate_tool_call_async(
                        engine = main_engine,
                        tokenizer = main_tokenizer,
                        question=entry["question"],
                        tools = entry["tools"],
                        prompt_passing_in_english = entry["prompt_passing_in_english"],
                )
                elif config.model in [Model.Local(LocalModel.Granite4_0HTiny), Model.Local(LocalModel.Granite4_0HSmall)]:
                    from src_py.granite4_backend import generate_tool_call_async
                    raw_output = await generate_tool_call_async(
                        engine = main_engine,
                        tokenizer = main_tokenizer,
                        question=entry["question"],
                        tools = entry["tools"],
                        prompt_passing_in_english = entry["prompt_passing_in_english"],
                )
                
                else:
                    raise NotImplementedError(f"Raw function call generation not implemented for model {config.model}.")
            except Exception as e:
                print(f"Error generating raw function call for index {entry['id']}: {e}")
                exit(1) # Todo: handle error properly
            # Each output entry has a signature of type GenerateRawAggregatedOutputEntry in src/tool/passes/pass_generate_raw.rs
            return {
                "id": entry["id"],
                "raw_output": raw_output,
                "file_name": entry["file_name"],
            }
    async def collect_all_raw_function_calls_async(entries: list[dict]) -> list[dict]:
        semaphore = asyncio.Semaphore(200)
        tasks = [collect_single_raw_function_call_async(entry, semaphore) for entry in entries]
        with open(aggregated_output_file_path, "w") as f:
            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                result = await coro
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"Generated raw function call {i}/{len(entries)}...")

    await collect_all_raw_function_calls_async(input_entries)
    # dispatch results to respective dataset files
    pass_generate_raw_dispatch_results(config)
    # finished processing raw function calls, deleting input file
    os.remove(aggregated_input_file_path)
    

    # generate raw function calls is done



    # The third pass is to convert the raw function calls to BFCL compatible function calls
    # For each raw result file, we call the rust function to convert it to BFCL compatible function calls
    print("----------PASS 3: PARSE FUNCTION CALLS----------")
    pass_parse_output(config)


    # The fourth pass is to post translate the function calls
    # We collect all parameter values that require translation
    # Then we call the python interface adapter to get translated parameter values
    # Then we replace the original parameter values with translated ones

    print("----------PASS 4: POST-TRANSLATE FUNCTION CALLS----------")
    pass_post_translate_prepare_aggregated_input(config)
    aggregated_input_file_path = pass_post_translate_aggregated_input_file_path(config)
    aggregated_output_file_path = pass_post_translate_aggregated_output_file_path(config)
    if os.path.exists(aggregated_output_file_path):
        pass_post_translate_dispatch_results(config)
    # Each entry has a signature of type PostTranslateAggregatedInputEntry in src/tool/passes/pass_post_translate.rs
    input_entries = load_json_lines_from_file(aggregated_input_file_path)

    async def collect_single_parameter_translation_async(entry: dict, semaphore) -> dict:
        # Create backend on demand
        global main_backend_created, main_client, main_engine, main_tokenizer, main_is_api
        if not main_backend_created:
            main_client, main_engine, main_tokenizer, main_is_api = create_backend(config.model)
            main_backend_created = True

        parameter_value = entry["parameter_value_to_translate"]
        if parameter_value.isascii():
            # no need to translate
            return {
                "original_parameter_value": parameter_value,
                "translated_parameter_value": parameter_value,
            }
        async with semaphore:
            try:
                if config.model in [Model.Api(ApiModel.Gpt5), Model.Api(ApiModel.Gpt5Mini), Model.Api(ApiModel.Gpt5Nano)]:
                    from src_py.gpt5_backend import translate_tool_parameter_async
                    translated_value = await translate_tool_parameter_async(
                        model_name = config.model.to_string(),
                        client = main_client,
                        parameter_value = parameter_value,
                    )
                elif config.model in [Model.Api(ApiModel.DeepSeek)]:
                    from src_py.deepseek_backend import translate_tool_answer_async
                    translated_value = await translate_tool_answer_async(
                        model_name = config.model.to_string(),
                        client = main_client,
                        parameter_value = parameter_value,
                    )
                elif config.model in [Model.Local(LocalModel.Qwen3_8B), Model.Local(LocalModel.Qwen3_14B), Model.Local(LocalModel.Qwen3_32B), Model.Local(LocalModel.Qwen3_30bA3b), Model.Local(LocalModel.Qwen3Next80bA3b)]:
                    from src_py.qwen3_backend import translate_tool_parameter_async
                    translated_value = await translate_tool_parameter_async(
                        engine = main_engine,
                        tokenizer = main_tokenizer,
                        parameter_value = parameter_value,
                    )
                elif config.model in [Model.Local(LocalModel.Llama3_1_8B), Model.Local(LocalModel.Llama3_1_70B)]:
                    from src_py.llama3_1_backend import translate_tool_parameter_async
                    translated_value = await translate_tool_parameter_async(
                        engine = main_engine,
                        tokenizer = main_tokenizer,
                        parameter_value = parameter_value,
                    )
                elif config.model in [Model.Local(LocalModel.Granite4_0HTiny), Model.Local(LocalModel.Granite4_0HSmall)]:
                    from src_py.granite4_backend import translate_tool_parameter_async
                    translated_value = await translate_tool_parameter_async(
                        engine = main_engine,
                        tokenizer = main_tokenizer,
                        parameter_value = parameter_value,
                    )
                else:
                    raise NotImplementedError(f"Parameter translation not implemented for model {config.model}.")
            except Exception as e:
                print(f"Error translating parameter index {entry['id']}: {e}")
                exit(1) # Todo: handle error properly
            # Each output entry has a signature of type PostTranslateAggregatedOutputEntry in src/tool/passes/pass_post_translate.rs
            return {
                "original_parameter_value": parameter_value,
                "translated_parameter_value": translated_value,
            }
    async def collect_all_parameter_translations_async(entries: list[dict]) -> list[dict]:
        semaphore = asyncio.Semaphore(200)
        tasks = [collect_single_parameter_translation_async(entry, semaphore) for entry in entries]
        with open(aggregated_output_file_path, "w") as f:
            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                result = await coro
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"Translated {i}/{len(entries)} parameters...")

    await collect_all_parameter_translations_async(input_entries)
    # dispatch results to respective dataset files
    pass_post_translate_dispatch_results(config)
    # finished processing post translation, deleting input file
    os.remove(aggregated_input_file_path)
    

    # The fifth pass is to evaluate the BFCL function calls
    # For each BFCL function call file, we call the rust function to evaluate it
    print("----------PASS 5: EVALUATE FUNCTION CALLS----------")
    pass_evaluate(config)
    # We can remove the scoring pass

    # The sixth pass is to categorize errors. 
    # In the first sub-pass, invalid parameter errors are collected. Other errors are ignored. No file is written to.
    # In the second sub-pass, all invalid parameter errors are categorized either through the cache or through gpt5.
    # Finally, we dispatch invalid parameter errors and determine other errors.
    print("----------PASS 6: CATEGORIZE PARAMETER VALUE MISMATCHES----------")
    pass_categorize_prepare_aggregated_input(config)
    aggregated_input_file_path = pass_categorize_aggregated_input_file_path(config)
    aggregated_output_file_path = pass_categorize_aggregated_output_file_path(config)
    if os.path.exists(aggregated_output_file_path):
        pass_categorize_dispatch_results(config)
    # Each entry has a signature of type CategorizeAggregatedInputEntry in src/tool/passes/pass_categorize.rs
    input_entries = load_json_lines_from_file(aggregated_input_file_path)
    # Create backend on demand
    if not assistant_backend_created and len(input_entries) > 0:
        assistant_client, assistant_engine, assistant_tokenizer, assistant_is_api = create_backend(Model.Api(ApiModel.Gpt5))
        assistant_backend_created = True

    semaphore = asyncio.Semaphore(200)
    async def collect_single_error_categorization_async(entry: dict) -> dict:
        async with semaphore:
            try:
                from src_py.gpt5_backend import categorize_parameter_value_async
                categorized_value = await categorize_parameter_value_async(
                    model_name = "gpt-5",
                    client = assistant_client,
                    actual_value = entry["actual_value"],
                    expected_values = entry["expected_values"],
                )
                print(f"Categorized actual_value: {entry['actual_value']} and expected_values: {entry['expected_values']} to category: {categorized_value}")
            except Exception as e:
                print(f"Error categorizing actual_value: {entry['actual_value']} and expected_values: {entry['expected_values']}: {e}")
                exit(1) # Todo: handle error properly
            return {
                "error_category": categorized_value,
                "actual_value": entry["actual_value"],
                "expected_values": entry["expected_values"],
            }
    async def collect_all_error_categorizations_async(entries: list[dict]) -> list[dict]:
        tasks = [collect_single_error_categorization_async(entry) for entry in entries]
        with open(aggregated_output_file_path, "w") as f:
            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                result = await coro
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"Categorized {i}/{len(entries)} errors...")
    await collect_all_error_categorizations_async(input_entries)
    # finished processing categorization, deleting input file
    os.remove(aggregated_input_file_path)
    # dispatch results to respective dataset files
    pass_categorize_dispatch_results(config)

    # The seventh pass is to generate the final report.
    print("----------PASS 7: GENERATE FINAL REPORT----------")
    pass_statistics(config)

if __name__ == "__main__":
    asyncio.run(main_async())