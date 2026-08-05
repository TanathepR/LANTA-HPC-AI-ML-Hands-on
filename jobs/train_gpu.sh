#!/bin/bash
# ---------------------------------------------------------------------------
# Lab 1b — MNIST training on 1 GPU (LANTA).
#
# Submit from YOUR personal workspace (see setup_user.sh), not the shared
# /project checkout:
#     cd ~/hpc-ai-workshop && sbatch jobs/train_gpu.sh
#
# GPU utilization is sampled every 5 seconds into logs/gpu-usage-<jobid>.csv
# so you can see how busy the GPU actually was during training.
# ---------------------------------------------------------------------------
#SBATCH --job-name=mnist-gpu
#SBATCH --partition=gpu              # LANTA GPU partition (A100)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
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

# --- Record GPU utilization in the background ------------------------------
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
           --format=csv -l 5 > "logs/gpu-usage-${SLURM_JOB_ID}.csv" &
GPU_LOG_PID=$!

# NOTE: MNIST must already be downloaded on the shared project (compute
# nodes have no internet) — see setup_project.sh.
python "${HPCAI_PROJECT_DIR}/scripts/train.py" --config configs/mnist_gpu.yaml

kill ${GPU_LOG_PID}
echo "GPU utilization log: logs/gpu-usage-${SLURM_JOB_ID}.csv"
