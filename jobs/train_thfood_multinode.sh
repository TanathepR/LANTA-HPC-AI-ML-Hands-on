#!/bin/bash
# ---------------------------------------------------------------------------
# Lab 2 — THFOOD-100 transfer learning across MULTIPLE NODES (LANTA).
#
# Same DistributedDataParallel (DDP) approach as train_thfood_multigpu.sh,
# extended to multiple nodes: one `torchrun` rendezvous spans all nodes in
# the Slurm allocation, launched via `srun` (one torchrun process per node,
# each of which then spawns --nproc_per_node local workers, one per GPU).
#
# Submit from YOUR personal workspace (see setup_user.sh), not the shared
# /project checkout:
#     cd ~/hpc-ai-workshop && sbatch jobs/train_thfood_multinode.sh
#
# Extra arguments are forwarded to train.py, same as the other job scripts:
#     sbatch jobs/train_thfood_multinode.sh --name thfood_2node --lr 0.0024
#
# NOTE: `training.batch_size` in the config is the PER-GPU batch size — the
# effective (global) batch size is batch_size * nodes * gpus-per-node.
# ---------------------------------------------------------------------------
#SBATCH --job-name=thfood-mnode
#SBATCH --partition=gpu              # LANTA GPU partition (A100)
#SBATCH --nodes=2                    # <-- adjust to how many nodes you want
#SBATCH --ntasks-per-node=1          # one torchrun launcher process per node
#SBATCH --gpus-per-node=4            # <-- adjust to how many GPUs per node
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

# Rendezvous point: the first node in the allocation coordinates process
# group setup for every torchrun instance across all nodes.
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=29500

echo "Job ${SLURM_JOB_ID}: ${SLURM_NNODES} node(s) x ${SLURM_GPUS_PER_NODE} GPU(s), master=${MASTER_ADDR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# NOTE: pretrained ImageNet weights must already be cached on the shared
# project (download them once on a login node — see setup_project.sh).
srun torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${SLURM_GPUS_PER_NODE}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${HPCAI_PROJECT_DIR}/scripts/train.py" --config configs/thfood_baseline.yaml "$@"
