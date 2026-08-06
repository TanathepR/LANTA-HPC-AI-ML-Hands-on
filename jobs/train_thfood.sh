#!/bin/bash
# ---------------------------------------------------------------------------
# Lab 2 — THFOOD-100 transfer learning on 1 GPU (LANTA).
#
# Submit from YOUR personal workspace (see setup_user.sh), not the shared
# /project checkout:
#     cd ~/hpc-ai-workshop && sbatch jobs/train_thfood.sh
#
# Extra arguments are forwarded to train.py, so you can run hyperparameter
# experiments WITHOUT editing any file:
#     sbatch jobs/train_thfood.sh --name thfood_lr001 --lr 0.001
#     sbatch jobs/train_thfood.sh --name thfood_bs128 --batch-size 128
# ---------------------------------------------------------------------------
#SBATCH --job-name=thfood
#SBATCH --partition=gpu              # LANTA GPU partition (A100)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
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
# 'hpc-ai' resolves by name because setup_user.sh registered the project's
# envs/ directory in your ~/.condarc -- the env itself still lives on
# /project, never in your home quota.
module purge
module load Mamba/23.11.0-0          # LANTA's conda distribution (adjust if needed)
conda activate hpc-ai

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Job ${SLURM_JOB_ID} on $(hostname):"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# NOTE: pretrained ImageNet weights must already be cached on the shared
# project (download them once on a login node — see setup_project.sh).
python "${HPCAI_PROJECT_DIR}/scripts/train.py" --config configs/thfood_baseline.yaml "$@"
