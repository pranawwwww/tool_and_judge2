
import os

from codebase_rs import *
os.environ['HF_HOME'] = "/work/nvme/bfdz/zluo8/huggingface"
from dotenv import load_dotenv
from utils import load_config_from_file
from utils import load_json_lines_from_file
from utils import combine_entries_to_pairs
import argparse
import subprocess
import time
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

# Start the first pass

print("Processing configuration: ", config)

# Determine alphabetical order for language codes
sorted_langs = sorted([config.lang1, config.lang2])
first_lang = sorted_langs[0]
second_lang = sorted_langs[1]




# Get or create backend with caching
model_name = config.model.value
display_model_name = get_model_directory_name(config.model)

# Get or create backend (batch size will be calculated automatically)
backend = get_or_create_backend(
    model_name=model_name,
    device="cuda",
    backend_type=backend_type,
    num_gpus=args.num_gpus
)

# Create model interface for model-specific behavior
model_interface = create_interface(model_name)
print(f"Using model interface: {model_interface.__class__.__name__}")

match config.experiment:
    case JudgeExperiment.PreferenceDirect(lang1=lang1, lang2=lang2):
        # load the datasets
        # load the model backend
        # collect the results in parallel
        # Process pairs for preference_direct
        for pairs, dataset_suffix in [
            (pairs_lang1_correct_lang2_incorrect, f"{first_lang}_correct_{second_lang}_incorrect"),
            (pairs_lang1_incorrect_lang2_correct, f"{first_lang}_incorrect_{second_lang}_correct"),
            (pairs_both_correct, f"{first_lang}_correct_{second_lang}_correct"),
            (pairs_both_incorrect, f"{first_lang}_incorrect_{second_lang}_incorrect")
        ]:
            output_file = f"judge/result/{display_model_name}/preferences_local_direct/{dataset_suffix}.jsonl"

            # Run async collection
            results = await collect_preference_local_direct_async(
                pairs=pairs,
                backend=backend,
                model_interface=model_interface,
                batch_size=8
            )

            # Write and sort results
            if results:
                append_and_rewrite_json_lines(output_file, results)    

    case JudgeExperiment.Perplexity(lang=lang):
        # Process individual entries for perplexity
        for entries, entry_suffix in [
            (entries_lang1_correct, f"{first_lang}_correct"),
            (entries_lang1_incorrect, f"{first_lang}_incorrect"),
            (entries_lang2_correct, f"{second_lang}_correct"),
            (entries_lang2_incorrect, f"{second_lang}_incorrect")
        ]:
            output_file = f"judge/result/{display_model_name}/perplexities_local/{entry_suffix}.jsonl"

            # Run async collection
            results = await collect_perplexity_local_async(
                entries=entries,
                backend=backend,
                model_interface=model_interface,
                batch_size=8
            )

            # Write and sort results
            if results:
                append_and_rewrite_json_lines(output_file, results)

    case _:
        print(f"Unknown result type: {config.result_type}")
        raise ValueError(f"Unknown result type: {config.result_type}")
print("Collected results for configuration: ", config)
