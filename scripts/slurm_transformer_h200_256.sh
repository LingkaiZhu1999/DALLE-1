#!/usr/bin/env bash
#SBATCH --job-name=dalle1-xfmr-h200
#SBATCH --partition=gpu-h200-141g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /scratch/work/zhul2/code/DALLE-1
mkdir -p runs/transformer_h200_256 runs/wandb

module load scicomp-python-env/2025.2

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$PWD/.uv-cache"
export WANDB_DIR="$PWD/runs/wandb"
export OMP_NUM_THREADS=16
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m dalle1.train_transformer \
  --config configs/transformer_h200_256.yaml \
  --dvae-checkpoint runs/dvae_256_b16/checkpoint-last.safetensors
