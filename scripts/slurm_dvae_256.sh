#!/usr/bin/env bash
#SBATCH --job-name=dalle1-dvae-h200
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /scratch/work/zhul2/code/DALLE-1
mkdir -p runs/dvae_256_b16 runs/wandb

source .venv/bin/activate

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$PWD/.uv-cache"
export WANDB_DIR="$PWD/runs/wandb"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

uv run python -m dalle1.train_dvae --config configs/dvae_h200_256.yaml
