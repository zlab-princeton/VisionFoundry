#!/usr/bin/env bash
set -euo pipefail

# Llama-3.2-11B-Vision-Instruct training (ms-swift)
# Fill in paths before running.

MODEL_PATH="/path/to/Llama-3.2-11B-Vision-Instruct"
DATASET_JSON="/path/to/annotations.json"
OUTPUT_DIR="/path/to/output/llama3_2_11b_vision"
LOG_DIR="/path/to/logs/llama3_2_11b_vision"

# Optional caches
# export MODELSCOPE_CACHE="/path/to/cache"
# export HF_HOME="/path/to/cache"
# export TRITON_CACHE_DIR="/path/to/triton_cache"
# export TMPDIR="/path/to/tmp"
# export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ----------------------
# Local single-node run
# ----------------------
# NPROC_PER_NODE=8 \
# swift sft \
#   --seed 42 \
#   --full_determinism false \
#   --data_seed 42 \
#   --dataset_shuffle false \
#   --train_dataloader_shuffle false \
#   --model "${MODEL_PATH}" \
#   --model_type llama3_2-11b-vision-instruct \
#   --train_type full \
#   --dataset "${DATASET_JSON}" \
#   --torch_dtype bfloat16 \
#   --attn_impl flash_attn \
#   --freeze_vision_tower false \
#   --freeze_mm_projector false \
#   --freeze_llm true \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 1 \
#   --learning_rate 5e-6 \
#   --vit_lr 5e-7 \
#   --gradient_accumulation_steps 16 \
#   --eval_steps -1 \
#   --save_steps 10000 \
#   --save_total_limit 1 \
#   --logging_steps 1 \
#   --max_length 2048 \
#   --output_dir "${OUTPUT_DIR}" \
#   --warmup_ratio 0.05 \
#   --dataloader_num_workers 4 \
#   --dataset_num_proc 8 \
#   --save_only_model false \
#   --deepspeed zero3 \
#   --gradient_checkpointing false \
#   --report_to tensorboard \
#   --logging_dir "${LOG_DIR}"

# ----------------------
# Slurm cluster run
# ----------------------
# sbatch <<'EOF'
# #!/bin/bash
# #SBATCH --job-name=llama_sft
# #SBATCH --partition=<partition>
# #SBATCH --gres=gpu:8
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=16
# #SBATCH --time=1:00:00
# #SBATCH --mem=256G
# #SBATCH --output=/path/to/slurm_logs/output_%j.log
# #SBATCH --error=/path/to/slurm_logs/error_%j.log
#
# module load anaconda3/<version>
# source /path/to/conda.sh
# conda activate <your_env>
#
# NPROC_PER_NODE=8 \
# swift sft \
#   --seed 42 \
#   --full_determinism false \
#   --data_seed 42 \
#   --dataset_shuffle false \
#   --train_dataloader_shuffle false \
#   --model "${MODEL_PATH}" \
#   --model_type llama3_2-11b-vision-instruct \
#   --train_type full \
#   --dataset "${DATASET_JSON}" \
#   --torch_dtype bfloat16 \
#   --attn_impl flash_attn \
#   --freeze_vision_tower false \
#   --freeze_mm_projector false \
#   --freeze_llm true \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 1 \
#   --learning_rate 5e-6 \
#   --vit_lr 5e-7 \
#   --gradient_accumulation_steps 16 \
#   --eval_steps -1 \
#   --save_steps 10000 \
#   --save_total_limit 1 \
#   --logging_steps 1 \
#   --max_length 2048 \
#   --output_dir "${OUTPUT_DIR}" \
#   --warmup_ratio 0.05 \
#   --dataloader_num_workers 4 \
#   --dataset_num_proc 8 \
#   --save_only_model false \
#   --deepspeed zero3 \
#   --gradient_checkpointing false \
#   --report_to tensorboard \
#   --logging_dir "${LOG_DIR}"
# EOF