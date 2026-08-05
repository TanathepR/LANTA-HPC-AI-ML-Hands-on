#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ONE-TIME setup of the SHARED project copy of this workshop (for a class
# with many students sharing one LANTA project account, e.g. tn999996).
#
# Home directories on LANTA have a small quota (e.g. 100 GB / 600k inodes),
# while /project has a much larger one (e.g. 30 TB / 300M inodes). This repo
# should be cloned directly onto /project, not into any student's $HOME:
#
#   git clone <repo-url> /project/tn999996-north/hpc-ai-workshop
#   cd /project/tn999996-north/hpc-ai-workshop
#   module load Mamba/23.11.0-0
#   bash setup_project.sh
#
# This builds the conda environment INSIDE this checkout (./envs/hpc-ai)
# instead of under $HOME/.conda, so the multi-GB PyTorch install never
# touches anyone's home quota. Every student then runs setup_user.sh once
# to get their own small, personal configs/jobs/checkpoints/outputs/logs
# under $HOME — see that script for details.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="${PROJECT_DIR}/envs"
ENV_PREFIX="${ENVS_DIR}/hpc-ai"

echo "=== HPC AI/ML Workshop — SHARED project setup ==="
echo "Project dir: ${PROJECT_DIR}"

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found in PATH."
    echo "On LANTA run:   module load Mamba/23.11.0-0"
    echo "then re-run:    bash setup_project.sh"
    exit 1
fi

if [ -d "${ENV_PREFIX}" ]; then
    echo "Environment already exists at ${ENV_PREFIX} — updating it."
    conda env update -p "${ENV_PREFIX}" -f environment.yml --prune
else
    echo "Creating conda environment at ${ENV_PREFIX} (this can take a few minutes)."
    conda env create -p "${ENV_PREFIX}" -f environment.yml
fi

# Register ${ENVS_DIR} as a place conda looks for named environments, so
# 'conda activate hpc-ai' finds it by name -- the environment files
# themselves still live entirely under /project. This edits only your own
# (tiny) ~/.condarc, not the environment itself.
conda config --append envs_dirs "${ENVS_DIR}"

echo
echo "Setup complete."
echo
echo "Next steps (project owner / instructor):"
echo "  1. Activate it and pre-download data on a LOGIN node (compute nodes"
echo "     often have no internet access):"
echo "       conda activate hpc-ai"
echo "       python datasets/download.py --dataset mnist --root ${PROJECT_DIR}/data"
echo "  2. Pre-download ImageNet weights for Lab 2 (also on a login node)."
echo "     TORCH_HOME must be set to the SHARED cache below (not the default"
echo "     \$HOME/.cache/torch) or students' jobs won't find these weights"
echo "     and will fail trying to download them from a no-internet compute"
echo "     node:"
echo "       export TORCH_HOME=${PROJECT_DIR}/cache/torch"
echo "       python -c \"import torchvision.models as m; \\"
echo "           m.resnet18(weights=m.ResNet18_Weights.IMAGENET1K_V1); \\"
echo "           m.mobilenet_v3_large(weights=m.MobileNet_V3_Large_Weights.IMAGENET1K_V2); \\"
echo "           m.efficientnet_b0(weights=m.EfficientNet_B0_Weights.IMAGENET1K_V1)\""
echo "  3. Obtain the full THFOOD-100 dataset once and place it under:"
echo "       ${PROJECT_DIR}/data/thfood100/{train,val,test}/<class_name>/*.jpg"
echo "  4. Make sure students can read (but not necessarily write) this"
echo "     directory, then have each of them run:"
echo "       bash ${PROJECT_DIR}/setup_user.sh"
