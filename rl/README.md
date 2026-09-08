# GRPO Training

These recipes run full-parameter multimodal GRPO with [verl](https://github.com/verl-project/verl), vLLM rollouts, and an OpenAI-compatible text judge. The judge compares the question, reference answer, and sampled answer and returns a binary reward.

## Environment

The released recipes were tested against verl `v0.4.1` (`8d9e350e`) with PyTorch 2.6, vLLM 0.8.5, Transformers 4.51.3, and FlashAttention 2.7.4.post1. Install verl following its documentation, then install the lightweight utilities here:

```bash
pip install -r rl/requirements.txt
```

Qwen2.5-VL and MiMo-VL use the standard full-parameter path. The Llama recipe additionally requires SDPA, variable-length Mllama cross-attention-mask handling, and visual/adapter/language optimizer groups. Apply the included compatibility patch before running Llama:

```bash
cd /path/to/verl
git checkout 8d9e350e
git apply /path/to/synthetic_data/rl/verl_v0.4.1_vlm.patch
pip install -e .
```

## Prepare Data

Convert the released SFT JSON to verl parquet. If the JSON contains paths from another machine, use both root arguments to relocate them:

```bash
python rl/prepare_data.py \
  --source /path/to/annotations.json \
  --output-dir /path/to/rl_data \
  --val-size 1000 \
  --source-image-root /old/dataset/root \
  --image-root /new/dataset/root
```

The resulting records contain `prompt`, `images`, `reward_model.ground_truth`, task metadata, and a deterministic train/validation split.

## Reward API

Set credentials only through environment variables; do not place keys in scripts:

```bash
export VISIONFOUNDRY_REWARD_API_KEY=<key>
export VISIONFOUNDRY_REWARD_BASE_URL=https://api.openai.com/v1
export VISIONFOUNDRY_REWARD_MODEL=Qwen2.5-3B
```

The reward accepts case and punctuation variations in `YES`/`NO`, retries transient or malformed responses, and stores valid judgments in SQLite. API failures are raised after the retry budget instead of being silently converted into incorrect zero rewards. The batch wrapper defaults to 32 concurrent requests; lower `VISIONFOUNDRY_REWARD_CONCURRENCY` if the provider rate-limits requests.

## Train

Each script runs locally or inside an existing Slurm allocation. Paths are supplied at launch:

```bash
export VERL_ROOT=/path/to/verl
export MODEL_PATH=/path/to/model
export DATA_DIR=/path/to/rl_data
export OUTPUT_DIR=/path/to/output
export CACHE_DIR=/path/to/cache
bash rl/train_qwen.sh
```

Use `train_mimo.sh` or `train_llama.sh` for the other backbones. All scripts use 8 GPUs by default, resume automatically from `OUTPUT_DIR`, and expose the main sweep variables as environment variables.

| Model | Batch | Learning rate | Rollouts | Response length | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-VL-3B-Instruct | 128 | `4e-6` | 4 | 128 | Full model, dynamic token batches |
| MiMo-VL-7B-SFT | 64 | `1e-6` | 4 | 128 | Full model, dynamic token batches |
| Llama-3.2-11B-Vision-Instruct | 32 | visual/adapter `1e-6`, language `0` | 4 | 32 | SDPA, TP=2, CPU offload |

The common settings are GRPO, `kl_loss_coef=0.01`, `low_var_kl`, one epoch, a 4096-token prompt limit, and one image per prompt. These are released settings rather than universal defaults; adjust global batch size and memory controls for different hardware.

For Slurm, invoke the same script from a site-specific submission wrapper instead of hard-coding cluster details into the repository:

```bash
#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=08:00:00

source /path/to/conda.sh
conda activate /path/to/env
export VERL_ROOT=/path/to/verl
export MODEL_PATH=/path/to/model
export DATA_DIR=/path/to/rl_data
export OUTPUT_DIR=/path/to/output
export CACHE_DIR=/path/to/cache
export VISIONFOUNDRY_REWARD_API_KEY=<provided-at-runtime>
bash /path/to/synthetic_data/rl/train_qwen.sh
```

verl writes sharded FSDP checkpoints. Convert the final actor checkpoint with the merger shipped by the same verl revision:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /path/to/output/global_step_N/actor \
  --target_dir /path/to/huggingface_checkpoint
```

