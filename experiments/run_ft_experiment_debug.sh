# !/bin/bash
export timestamp=$(date +'%Y-%m-%d_%H-%M-%S')

NUM_PROCESSES=1
EXPERIMENT_CONFIG=experiments/samedoc/configs/both_hops_samedoc.yaml

experiment_name=$(basename $EXPERIMENT_CONFIG .yaml) # e.g., "atomic+2hop"
run_name="${timestamp}_${SLURM_JOB_ID}_${experiment_name}"

random_port=$(shuf -i 20000-65535 -n 1)

WANDB_NAME="$run_name" accelerate launch \
--num_processes $NUM_PROCESSES \
--config_file latent_reasoning/fsdp_accelerate_config_debug.yaml \
--main_process_port $random_port latent_reasoning/train.py \
--config experiments/fully_synthetic/trl_config.yaml \
--output_dir "models/$run_name" \
--experiment_config $EXPERIMENT_CONFIG \
--use_peft_lora \
--lora_r=1