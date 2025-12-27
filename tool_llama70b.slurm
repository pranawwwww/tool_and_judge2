#!/bin/bash
#SBATCH --job-name=llama70b       # Job name
#SBATCH --output=z_llama70b_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=z_llama70b_%j.err         # Error log
#SBATCH --time=04:00:00                 # Max runtime (HH:MM:SS) - increased for 70B model
# #SBATCH --partition=gpu                  # Partition/queue with GPU
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=luozheng@usc.edu
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:8                     # Request 8 GPUs (H200 for 70B model)
#SBATCH --cpus-per-task=8                # CPU cores - increased for larger model
#SBATCH --mem=128G                       # RAM - increased for 70B model
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment

# Go to your project directory
cd /projects/bfdz/zluo8/tool_and_judge2
source activate_environment.sh
cd /projects/bfdz/zluo8/tool_and_judge2

# Set HF_HOME environment variable
export HF_HOME=/work/nvme/bfdz/zluo8/huggingface

# Run the Llama 3.1 70B experiments with Chinese, Hindi, and Igbo
# Uses 8 GPUs for tensor parallelism
python tool.py --config tool_config_llama70b.py --num-gpus 8

