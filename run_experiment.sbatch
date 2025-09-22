#!/bin/bash
#SBATCH --job-name=latent-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Load required modules (adjust based on your cluster)
# module load cuda/12.1
# module load python/3.10

# Activate virtual environment
source .venv/bin/activate

# Set up environment variables
export WANDB_PROJECT="latent_reasoning"
export HF_HOME="${SCRATCH}/huggingface"  # Use scratch for model cache

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the experiment
# Arguments: num_gpus config_path [additional_args]
./experiments/run_ft_experiment.sh 4 experiments/samedoc/configs/both_hops_samedoc_distractor_triplets.yaml