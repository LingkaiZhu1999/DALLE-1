#!/usr/bin/env bash
#SBATCH --job-name=dalle1-xfmr-probe
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /scratch/work/zhul2/code/DALLE-1
mkdir -p runs/transformer_probe runs/wandb

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$PWD/.uv-cache"
export WANDB_DIR="$PWD/runs/wandb"

uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_devices", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))
PY

uv run python -m dalle1.train_transformer \
  --config configs/transformer_probe.yaml \
  --dvae-checkpoint runs/dvae_probe/checkpoint-last.safetensors
