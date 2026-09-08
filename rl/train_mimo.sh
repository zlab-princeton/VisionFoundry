#!/usr/bin/env bash
set -euo pipefail

# MiMo-VL-7B-SFT full-parameter GRPO on one 8-GPU node.
: "${VERL_ROOT:?Set VERL_ROOT to a verl v0.4.1 checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to MiMo-VL-7B-SFT}"
: "${DATA_DIR:?Set DATA_DIR to the directory containing train.parquet and val.parquet}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${VISIONFOUNDRY_REWARD_API_KEY:?Set VISIONFOUNDRY_REWARD_API_KEY}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
N_GPUS=${N_GPUS:-8}
CACHE_DIR=${CACHE_DIR:-"${HOME}/.cache/visionfoundry"}
RUN_NAME=${RUN_NAME:-mimo_vl_7b_visionfoundry_grpo}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
TRAIN_LR=${TRAIN_LR:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

export HF_HOME=${HF_HOME:-"${CACHE_DIR}/huggingface"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"${CACHE_DIR}/triton"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${CACHE_DIR}/xdg"}
export VISIONFOUNDRY_REWARD_MODEL=${VISIONFOUNDRY_REWARD_MODEL:-Qwen2.5-3B}
export VISIONFOUNDRY_REWARD_BASE_URL=${VISIONFOUNDRY_REWARD_BASE_URL:-https://api.openai.com/v1}
export VISIONFOUNDRY_REWARD_CACHE_DB=${VISIONFOUNDRY_REWARD_CACHE_DB:-"${CACHE_DIR}/${RUN_NAME}.sqlite"}
export VISIONFOUNDRY_REWARD_TIMEOUT=${VISIONFOUNDRY_REWARD_TIMEOUT:-90}
export VISIONFOUNDRY_REWARD_MAX_RETRIES=${VISIONFOUNDRY_REWARD_MAX_RETRIES:-10}
export VISIONFOUNDRY_REWARD_MAX_TOKENS=${VISIONFOUNDRY_REWARD_MAX_TOKENS:-64}
export VISIONFOUNDRY_REWARD_CONCURRENCY=${VISIONFOUNDRY_REWARD_CONCURRENCY:-32}
export VLLM_USE_V1=${VLLM_USE_V1:-0}
mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}"

cd "${VERL_ROOT}"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.image_key=images \
  +data.dataloader_num_workers=0 \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=4096 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${TRAIN_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt='{image: 1, video: 0}' \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs='{min_pixels: 200704, max_pixels: 1003520}' \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  reward_model.reward_manager=batch \
  reward_model.launch_reward_fn_async=True \
  +reward_model.reward_kwargs.concurrency="${VISIONFOUNDRY_REWARD_CONCURRENCY}" \
  custom_reward_function.path="${SCRIPT_DIR}/batch_text_judge_reward.py" \
  custom_reward_function.name=compute_score \
  ray_init.num_cpus="${RAY_NUM_CPUS:-16}" \
  trainer.logger=console \
  trainer.project_name=visionfoundry_rl \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq="${SAVE_FREQ:-20}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.resume_mode=auto

