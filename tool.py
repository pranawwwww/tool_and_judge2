import argparse
import asyncio
import subprocess
import time
import signal
import sys

from load_configs_from_file import load_configs_from_file



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
configs = load_configs_from_file(args.config, "configs")

# Set up signal handlers for fast Ctrl+C interruption
async def run_with_cancellation():
    """Run the main task with proper cancellation on Ctrl+C."""
    main_task = asyncio.create_task(codebase_rs.tool_run_async(configs, args.num_gpus))

    def signal_handler(_signum, _frame):
        print("\n⚠️  Ctrl+C detected! Cancelling all tasks...", flush=True)
        main_task.cancel()
        # Cancel all other running tasks
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        await main_task
    except asyncio.CancelledError:
        print("✓ Tasks cancelled successfully.", flush=True)
        sys.exit(130)  # Standard exit code for Ctrl+C

asyncio.run(run_with_cancellation())



# asyncio.run(process_all_configs())