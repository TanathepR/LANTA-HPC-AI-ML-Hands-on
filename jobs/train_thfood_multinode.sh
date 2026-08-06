#!/bin/bash
# ---------------------------------------------------------------------------
# Lab 2 — THFOOD-100 transfer learning across MULTIPLE NODES (LANTA).
#
# Same DistributedDataParallel (DDP) approach as train_thfood_multigpu.sh,
# extended to multiple nodes. Launches ONE PROCESS PER GPU DIRECTLY via
# `srun` (Slurm assigns each process its rank through SLURM_PROCID /
# SLURM_LOCALID) instead of going through `torchrun`'s own rendezvous —
# this matches LANTA's own documented pattern for multi-GPU/multi-node
# PyTorch jobs. torchrun's separate rendezvous handshake (a second TCP
# connection distinct from the actual training process group) timed out
# connecting across nodes on LANTA's network even though the training
# process group itself (the same underlying mechanism, `env://` init)
# works fine — so this launch style avoids that failure mode.
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
#SBATCH --ntasks-per-node=4          # one task per GPU (must match --gpus-per-node)
#SBATCH --gpus-per-node=4            # <-- adjust to how many GPUs per node
#SBATCH --cpus-per-task=16           # (cores per node) / (GPUs per node)
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# -----------------------------------------------------------------------
# Rendezvous point for torch.distributed's process group (env:// init).
# Every srun-launched process reads these three variables directly;
# trainer.utils.init_distributed does the rest (SLURM_PROCID -> rank,
# SLURM_LOCALID -> local rank).
# -----------------------------------------------------------------------
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_ADDR
echo "MASTER_ADDR=${MASTER_ADDR}"

# Derived from the job ID (not a fixed port) so two concurrent jobs on the
# shared cluster never collide trying to bind/connect to the same port —
# a fixed port here previously caused every rank, including the master
# node connecting to itself, to time out.
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MASTER_PORT
echo "MASTER_PORT=${MASTER_PORT}"

WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export WORLD_SIZE
echo "WORLD_SIZE=${WORLD_SIZE}"

echo "Job ${SLURM_JOB_ID}: ${SLURM_NNODES} node(s) x ${SLURM_GPUS_PER_NODE} GPU(s), master=${MASTER_ADDR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# NOTE: pretrained ImageNet weights must already be cached on the shared
# project (download them once on a login node — see setup_project.sh).
srun python "${HPCAI_PROJECT_DIR}/scripts/train.py" --config configs/thfood_baseline.yaml "$@"
