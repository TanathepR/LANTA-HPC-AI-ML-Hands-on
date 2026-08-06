#!/bin/bash
# ---------------------------------------------------------------------------
# Lab 1a — MNIST training on CPU only (LANTA).
#
# Submit from YOUR personal workspace (see setup_user.sh), not the shared
# /project checkout:
#     cd ~/hpc-ai-workshop && sbatch jobs/train_cpu.sh
#
# Compare the resulting epoch time and images/sec with the GPU run
# (jobs/train_gpu.sh) — the code and hyperparameters are identical.
# ---------------------------------------------------------------------------
#SBATCH --job-name=mnist-cpu
#SBATCH --partition=compute          # LANTA CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

# Let PyTorch's math libraries use every CPU core Slurm gave us.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Job ${SLURM_JOB_ID} on $(hostname): ${SLURM_CPUS_PER_TASK} CPU cores"

# NOTE: MNIST must already be downloaded on the shared project (compute
# nodes have no internet) — see setup_project.sh.
python "${HPCAI_PROJECT_DIR}/scripts/train.py" --config configs/mnist_cpu.yaml
