import argparse
import asyncio
import subprocess
import time

from src_py.utils import load_config_from_file
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
    help="Path to a Python file containing the 'configs'"
)
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for local inference (default: 1)"
)
args = parser.parse_args()

# Load configs from specified file
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
        result = subprocess.run(["maturin", "develop"], check=True)
        print("Installed Rust extension successfully.")
        time.sleep(2)  # Give some time for the build to complete
    finally:
        # Release lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        print("Released build lock.")

from codebase_rs import *

print(f"Loading configs from: {args.config}")
config = load_config_from_file(args.config, "configs")
print("Processing configuration: ", config)

# api model needs client and vllm model needs an engine and a tokenizer


# Acquire model-level lock, so that all aggregated files can be free of race condition, and is safe to read

# The first pass is pre-translation
# aggregate all questions across datasets
# We can create a dataset file that contains questions only for each dataset file
# aggregate and extract questions can be done at the same time
# This involves calling a rust function
# Then we have the question only dataset file. Its path can be retrieved from Rust code.
# Then we get the python array object from reading the file

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

# The third pass is to convert the raw function calls to BFCL compatible function calls
# For each raw result file, we call the rust function to convert it to BFCL compatible function calls

# The fourth pass is to post translate the function calls
# We collect all parameter values that require translation
# Then we call the python interface adapter to get translated parameter values
# Then we replace the original parameter values with translated ones


# The fifth pass is to evaluate the BFCL function calls
# For each BFCL function call file, we call the rust function to evaluate it

# We can remove the scoring pass

# The sixth pass is to categorize errors. 
# In the first sub-pass, invalid parameter errors are collected. Other errors are ignored. No file is written to.
# In the second sub-pass, all invalid parameter errors are categorized either through the cache or through gpt5.
# Finally, we dispatch invalid parameter errors and determine other errors.

# The seventh pass is to generate the final report.



