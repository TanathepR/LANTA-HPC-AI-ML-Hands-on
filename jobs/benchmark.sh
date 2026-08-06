#!/bin/bash
# ---------------------------------------------------------------------------
# Benchmark all Lab 2 models on 1 GPU (LANTA).
#
# Submit from YOUR personal workspace (see setup_user.sh), not the shared
# /project checkout:
#     cd ~/hpc-ai-workshop && sbatch jobs/benchmark.sh
#
# Results are printed to the Slurm log and saved as
# outputs/benchmark_<model>.json for each model.
# ---------------------------------------------------------------------------
#SBATCH --job-name=benchmark
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

# Compare the three transfer-learning backbones (random weights — speed
# does not depend on weight values, so no download is needed).
for model in resnet18 mobilenetv3 efficientnet_b0; do
    python "${HPCAI_PROJECT_DIR}/scripts/benchmark.py" --model "${model}" --num-classes 100 --batch-size 64
done

# The Lab 1 model, for reference.
python "${HPCAI_PROJECT_DIR}/scripts/benchmark.py" --model lenet --num-classes 10 --img-size 28 --in-channels 1
