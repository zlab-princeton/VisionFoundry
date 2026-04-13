#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-VL-3B-Instruct training (ms-swift)
# Fill in paths before running.

MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
DATASET_JSON="/path/to/annotations.json"
OUTPUT_DIR="/path/to/output/qwen2_5_vl_3b"
LOG_DIR="/path/to/logs/qwen2_5_vl_3b"

# Optional caches
# export MODELSCOPE_CACHE="/path/to/cache"
# export HF_HOME="/path/to/cache"
# export TRITON_CACHE_DIR="/path/to/triton_cache"
# export TMPDIR="/path/to/tmp"

# ----------------------
# Local single-node run
# ----------------------
# NPROC_PER_NODE=8 \
# MAX_PIXELS=1003520 \
# swift sft \
#   --seed 42 \
#   --full_determinism false \
#   --data_seed 42 \
#   --dataset_shuffle false \
#   --train_dataloader_shuffle false \
#   --model "${MODEL_PATH}" \
#   --model_type qwen2_5_vl \
#   --train_type full \
#   --dataset "${DATASET_JSON}" \
#   --torch_dtype bfloat16 \
#   --attn_impl flash_attn \
#   --freeze_vit false \
#   --freeze_llm false \
#   --freeze_aligner false \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 4 \
#   --learning_rate 5e-6 \
#   --vit_lr 5e-7 \
#   --gradient_accumulation_steps 4 \
#   --eval_steps -1 \
#   --save_steps 10000 \
#   --save_total_limit 1 \
#   --logging_steps 1 \
#   --max_length 8192 \
#   --output_dir "${OUTPUT_DIR}" \
#   --warmup_ratio 0.05 \
#   --dataloader_num_workers 4 \
#   --dataset_num_proc 8 \
#   --save_only_model true \
#   --deepspeed zero2 \
#   --report_to tensorboard \
#   --logging_dir "${LOG_DIR}"

# ----------------------
# Slurm cluster run
# ----------------------
# sbatch <<'EOF'
# #!/bin/bash
# #SBATCH --job-name=qwen_sft
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
# MAX_PIXELS=1003520 \
# swift sft \
#   --seed 42 \
#   --full_determinism false \
#   --data_seed 42 \
#   --dataset_shuffle false \
#   --train_dataloader_shuffle false \
#   --model "${MODEL_PATH}" \
#   --model_type qwen2_5_vl \
#   --train_type full \
#   --dataset "${DATASET_JSON}" \
#   --torch_dtype bfloat16 \
#   --attn_impl flash_attn \
#   --freeze_vit false \
#   --freeze_llm false \
#   --freeze_aligner false \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 4 \
#   --learning_rate 5e-6 \
#   --vit_lr 5e-7 \
#   --gradient_accumulation_steps 4 \
#   --eval_steps -1 \
#   --save_steps 10000 \
#   --save_total_limit 1 \
#   --logging_steps 1 \
#   --max_length 8192 \
#   --output_dir "${OUTPUT_DIR}" \
#   --warmup_ratio 0.05 \
#   --dataloader_num_workers 4 \
#   --dataset_num_proc 8 \
#   --save_only_model true \
#   --deepspeed zero2 \
#   --report_to tensorboard \
#   --logging_dir "${LOG_DIR}"
# EOF