#!/bin/bash
#SBATCH --job-name=llama8b       # Job name
#SBATCH --output=z_llama8b_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=z_llama8b_%j.err         # Error log
#SBATCH --time=02:00:00                 # Max runtime (HH:MM:SS) - increased for full experiments
# #SBATCH --partition=gpu                  # Partition/queue with GPU
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=luozheng@usc.edu
# #SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:1                     # Request 1 GPU (Llama 8B fits on 1 GPU)
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --mem=64G                        # RAM - increased for safety
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment

# Go to your project directory
cd /projects/bfdz/zluo8/tool_and_judge2
source activate_environment.sh
cd /projects/bfdz/zluo8/tool_and_judge2

# Set HF_HOME environment variable
export HF_HOME=/work/nvme/bfdz/zluo8/huggingface

# Run the Llama 3.1 8B experiments with Chinese, Hindi, and Igbo
python tool.py --config tool_config_llama8b.py --num-gpus 1

