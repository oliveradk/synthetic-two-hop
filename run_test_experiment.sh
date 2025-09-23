#!/bin/bash
# Test script for quick iteration on the both_hops_samedoc_distractor_triplets experiment

export timestamp=$(date +'%Y-%m-%d_%H-%M-%S')

# Default to 1 GPU for test run, can override with first argument
NUM_PROCESSES=${1:-1}

# Test config paths
EXPERIMENT_CONFIG="experiments/samedoc/configs/test_both_hops_samedoc_distractor_triplets.yaml"
TRL_CONFIG="experiments/samedoc/configs/test_trl_lora_config.yaml"  # Using LoRA config

experiment_name="test_samedoc_distractor_lora"
run_name="${timestamp}_${experiment_name}"

# Use a random port to avoid conflicts
random_port=$(shuf -i 20000-65535 -n 1)

echo "Starting test run: $run_name"
echo "Using config: $EXPERIMENT_CONFIG"
echo "Training for 1 step with minimal data..."

# Run the training
WANDB_MODE="disabled" \
WANDB_NAME="$run_name" \
accelerate launch \
--num_processes $NUM_PROCESSES \
--config_file latent_reasoning/fsdp_accelerate_config.yaml \
--main_process_port $random_port \
latent_reasoning/train.py \
--config $TRL_CONFIG \
--output_dir "models/$run_name" \
--experiment_config $EXPERIMENT_CONFIG

echo "Test run completed: models/$run_name"