
## Fully-Synthetic: Same-Document Two Stage


Stage 1: Train on atomic facts
```bash
export WANDB_TAGS="samedoc, distractor_triplets, facts_stage"
./experiments/run_ft_experiment.sh 4 experiments/samedoc_two_stage/both_hops_samedoc_distractor_triplets_facts.yaml \
    --seed 42 --lr_scheduler_type "cosine" --warmup_ratio 0.05 --save_strategy "epoch"
```

Find Weights:
- `models/<timestamp>_<job_id>_both_hops_samedoc_distractor_triplets_facts/checkpoint-<step>/pytorch_model.bin`
- or `models/<timestamp>_<job_id>_both_hops_samedoc_distractor_triplets_facts/checkpoint-<step>/model.safetensors`

Stage 2: Train on 2-hop reasoning with Stage 1 weights
```bash
export WANDB_TAGS="samedoc, distractor_triplets, 2hop_stage"
./experiments/run_ft_experiment.sh 4 experiments/samedoc_two_stage/both_hops_samedoc_distractor_triplets_facts_2hop.yaml \
    --seed 42 --lr_scheduler_type "cosine" --warmup_ratio 0.05 \
    --load_weights_from "models/YOUR_STAGE1_RUN/checkpoint-XXX/pytorch_model.bin"
```

Replace `YOUR_STAGE1_RUN` and `XXX` with the actual paths from your Stage 1 training.
