#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ONE-TIME setup of YOUR personal workspace, for use with a shared /project
# copy of this workshop (see setup_project.sh). Run once, from anywhere:
#
#   bash /project/tn999996-north/training_ai_ml/setup_user.sh
#
# Optional arguments:
#   $1 = path to the shared project checkout (default: where this script
#        lives, or $HPCAI_PROJECT_DIR if set)
#   $2 = where to create your personal workspace (default: ~/hpc-ai-workshop)
#
# This does NOT clone the repo or install any conda packages into $HOME —
# it only copies the small, personal parts you are meant to edit (configs/
# and jobs/) and creates empty checkpoints/outputs/logs directories for your
# own runs. The code, conda environment, and datasets stay on /project and
# are never duplicated into your (much smaller) home quota.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${1:-${HPCAI_PROJECT_DIR:-$SCRIPT_DIR}}"
WORKSPACE="${2:-$HOME/hpc-ai-workshop}"

echo "=== HPC AI/ML Workshop — personal workspace setup ==="
echo "Project dir : ${PROJECT_DIR}"
echo "Workspace   : ${WORKSPACE}"

if [ ! -d "${PROJECT_DIR}/scripts" ] || [ ! -d "${PROJECT_DIR}/configs" ]; then
    echo "ERROR: '${PROJECT_DIR}' does not look like the workshop repo (no scripts/ or configs/ found)."
    echo "Pass the shared project checkout path as the first argument, e.g.:"
    echo "  bash setup_user.sh /project/tn999996-north/training_ai_ml"
    exit 1
fi

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found in PATH."
    echo "On LANTA run:   module load Mamba/23.11.0-0"
    echo "then re-run:    bash setup_user.sh"
    exit 1
fi

# Register the shared project's envs/ directory as a place conda looks for
# named environments, so 'conda activate hpc-ai' finds it by name from
# anywhere -- the environment itself stays on /project; this only edits
# your own (tiny) ~/.condarc.
conda config --append envs_dirs "${PROJECT_DIR}/envs"

mkdir -p "${WORKSPACE}"
if [ -d "${WORKSPACE}/configs" ]; then
    echo "configs/ already exists in your workspace — leaving it as is."
else
    cp -r "${PROJECT_DIR}/configs" "${WORKSPACE}/"
    # Jobs run with this workspace (not the /project checkout) as the
    # working directory, so a relative "root: ./data/..." would resolve to
    # a nonexistent folder under $HOME instead of the shared dataset on
    # /project. Rewrite it to an absolute path pointing at the shared data.
    sed -i "s#root: \./data#root: ${PROJECT_DIR}/data#" "${WORKSPACE}"/configs/*.yaml
fi
[ -d "${WORKSPACE}/jobs" ] || cp -r "${PROJECT_DIR}/jobs" "${WORKSPACE}/"
mkdir -p "${WORKSPACE}/checkpoints" "${WORKSPACE}/outputs" "${WORKSPACE}/logs"

# link script file in project to user
ln -sfn "${PROJECT_DIR}/scripts" "${WORKSPACE}/scripts"

cat > "${WORKSPACE}/project.env" <<EOF
# Written by setup_user.sh — points your job scripts at the shared
# /project checkout (code + conda environment) and the shared PyTorch
# pretrained-weight cache. Safe to re-source; only edit this if the
# project ever moves to a different path. Always regenerated when you
# re-run setup_user.sh, so it never falls out of date.
export HPCAI_PROJECT_DIR="${PROJECT_DIR}"
# PyTorch's default weight cache is per-user (\$HOME/.cache/torch), so a
# pretrained model download by one student would be invisible to everyone
# else and would fail outright on a no-internet compute node. Pointing
# TORCH_HOME at the shared project means the one download the instructor
# does on a login node (see setup_project.sh) benefits every student.
export TORCH_HOME="${PROJECT_DIR}/cache/torch"
EOF

echo
echo "Done. Your personal workspace is ready at: ${WORKSPACE}"
echo "'conda activate hpc-ai' now works from anywhere -- the environment"
echo "itself still lives entirely on /project, not in your home quota."
echo
echo "  configs/      -> your own copies of the YAML configs, edit freely"
echo "  jobs/         -> your own copies of the Slurm scripts, edit freely"
echo "  checkpoints/  -> keep your best checkpoints here (survives re-runs)"
echo "  outputs/      -> per-run results (checkpoints, metrics, config copy)"
echo "  logs/         -> TensorBoard event files + Slurm logs"
echo "  scripts/      -> link shared script files to your home (train, benchmark, evaluate, predict, export)"
echo
echo "Next steps:"
echo "  cd ${WORKSPACE}"
echo "  # 1. edit jobs/*.sh: set #SBATCH --account=ltXXXXXX to your LANTA account"
echo "  # 2. submit a job (dataset.root in configs/*.yaml already points at"
echo "  #    the shared /project data, no edit needed)"
echo "  sbatch jobs/train_cpu.sh"
