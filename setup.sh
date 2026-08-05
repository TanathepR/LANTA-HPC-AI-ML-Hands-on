#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time environment setup for the HPC AI/ML workshop.
#
# Usage (from the repository root):
#   bash setup.sh
#
# On LANTA, load the conda distribution first:
#   module load Mamba/23.11.0-0
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="hpc-ai"

echo "=== HPC AI/ML Workshop — environment setup ==="

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found in PATH."
    echo "On LANTA run:   module load Mamba/23.11.0-0"
    echo "then re-run:    bash setup.sh"
    exit 1
fi

# Create the environment, or update it if it already exists.
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Environment '${ENV_NAME}' already exists — updating it."
    conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
    echo "Creating conda environment '${ENV_NAME}' (this can take a few minutes)."
    conda env create -f environment.yml
fi

echo
echo "Setup complete."
echo
echo "Next steps:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. Pre-download data on a login node (compute nodes may have no internet):"
echo "       python datasets/download.py --dataset mnist --root ./data"
echo "  3. Pre-download ImageNet weights for Lab 2 (also on a login node):"
echo "       python -c \"import torchvision.models as m; m.resnet18(weights=m.ResNet18_Weights.IMAGENET1K_V1)\""
