#!/bin/bash
#SBATCH --job-name=judge3       # Job name
#SBATCH --output=z_judge3_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=z_judge3_%j.err         # Error log
#SBATCH --time=00:30:00                 # Max runtime (HH:MM:SS)
# #SBATCH --partition=gpu                  # Partition/queue with GPU
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=luozheng@usc.edu
# #SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --mem=32G                        # RAM
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment

# Go to your project directory
# cd /u/zluo8/qwen
cd /projects/bfdz/zluo8/tool_and_judge2
source activate_environment.sh
cd /projects/bfdz/zluo8/tool_and_judge2
# Set HF_HOME environment variable
export HF_HOME=/work/nvme/bfdz/zluo8/huggingface
# Convert the notebook to a .py script
# jupyter nbconvert --to script test_toolrl.ipynb

# Run the training script

python judge.py --config judge_config_slurm3.py --num-gpus 1