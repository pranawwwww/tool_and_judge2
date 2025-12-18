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

# Run maturin develop to build and install the Rust extension
print("Building Rust extension with maturin develop...")
result = subprocess.run(["maturin", "develop"], check=True)

# Now import and use the module
time.sleep(2)  # Give some time for the build to complete
import codebase_rs

print(f"Loading configs from: {args.config}")
configs = load_config_from_file(args.config, "configs")

# Run the async task
# Note: Ctrl+C handling is done in Rust (tool_run.rs) using the ctrlc crate
# The Rust handler will gracefully shut down between configs
asyncio.run(codebase_rs.tool_run_async(configs, args.num_gpus))


# Refactor idea:
# Separate python and rust code completely.
# One pass runs python and the other runs rust.
# Still needs to take advantage of rust's model interface.

# Python side: string input, string output

# choose backend implementation based on model name