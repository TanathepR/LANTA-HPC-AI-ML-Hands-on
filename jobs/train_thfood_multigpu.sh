#!/bin/bash
# ---------------------------------------------------------------------------
# Lab 2 — THFOOD-100 transfer learning on MULTIPLE GPUs, 1 node (LANTA).
#
# Uses DistributedDataParallel (DDP) via `torchrun`: each GPU trains on its
# own shard of the data and gradients are averaged across GPUs every step —
# same math as single-GPU training, just more images/sec. No source code
# or config changes are needed versus jobs/train_thfood.sh.
#
# Submit from YOUR personal workspace (see setup_user.sh), not the shared
# /project checkout:
#     cd ~/hpc-ai-workshop && sbatch jobs/train_thfood_multigpu.sh
#
# Extra arguments are forwarded to train.py, same as the single-GPU script:
#     sbatch jobs/train_thfood_multigpu.sh --name thfood_4gpu --lr 0.0012
#
# NOTE: `training.batch_size` in the config is the PER-GPU batch size —
# the effective (global) batch size is batch_size * --gpus-per-node. If you
# compare against a single-GPU run, expect to scale the learning rate up
# accordingly (a common starting point is linear scaling with batch size).
# ---------------------------------------------------------------------------
#SBATCH --job-name=thfood-mgpu
#SBATCH --partition=gpu              # LANTA GPU partition (A100)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4            # <-- adjust to how many GPUs you want
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --account=tn999996           # <-- replace with your LANTA project account
#SBATCH --output=logs/slurm-%x-%j.out

# Run from the directory the job was submitted from (your personal
# workspace — configs/, outputs/, logs/, checkpoints/ all live here).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs checkpoints outputs

# project.env is written once by setup_user.sh and points HPCAI_PROJECT_DIR
# at the shared /project checkout: code, conda environment, and datasets
# all live there so they never touch your (much smaller) home quota.
source ./project.env

# --- Activate the shared conda environment ---------------------------------
module purge
module load Mamba/23.11.0-0          # LANTA's conda distribution (adjust if needed)
conda activate hpc-ai

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE))

echo "Job ${SLURM_JOB_ID} on $(hostname): ${SLURM_GPUS_PER_NODE} GPU(s)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# NOTE: pretrained ImageNet weights must already be cached on the shared
# project (download them once on a login node — see setup_project.sh).
#
# --standalone: single-node rendezvous, no MASTER_ADDR/MASTER_PORT needed.
torchrun --standalone --nproc_per_node="${SLURM_GPUS_PER_NODE}" \
    "${HPCAI_PROJECT_DIR}/scripts/train.py" --config configs/thfood_baseline.yaml "$@"
