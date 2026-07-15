#!/usr/bin/env bash
#SBATCH --job-name=dalle1-dvae-h200
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /scratch/work/zhul2/code/DALLE-1
mkdir -p runs/dvae_256_paper runs/wandb

module load scicomp-python-env/2025.2
source .venv/bin/activate

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$PWD/.uv-cache"
export WANDB_DIR="$PWD/runs/wandb"
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
if [[ "$GPUS_PER_NODE" == *","* ]]; then
  GPUS_PER_NODE="$(tr ',' '\n' <<<"$GPUS_PER_NODE" | wc -l)"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m dalle1.train_dvae \
  --config configs/dvae_h200_256.yaml
